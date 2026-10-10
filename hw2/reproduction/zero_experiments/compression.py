"""Local magnitude pruning and executable channel Tucker-2 convolution."""
import hashlib
import json
from pathlib import Path
from dataclasses import replace
import torch
from torch import nn
from .config import Config
from .models import create_model
from .data import make_loaders
from .ranks import kernel_ranks
from .training import epoch_pass
from .runtime import configure, write_csv, write_json, environment

@torch.no_grad()
def magnitude_prune(model, original, keep):
    if not 0 <= keep <= 1:
        raise ValueError('keep must be in [0,1]')
    model.load_state_dict(original)
    kept, total, nonzero = 0, 0, 0
    layers = []
    for name, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        flat = original[name+'.weight'].flatten()
        n = int(keep*flat.numel())
        indices = torch.argsort(flat.abs(), descending=True, stable=True)[:n]
        pruned = torch.zeros_like(flat)
        pruned[indices] = flat[indices]
        layer.weight.copy_(pruned.reshape_as(layer.weight))
        kept += n
        total += flat.numel()
        nonzero += int(torch.count_nonzero(pruned))
        layers.append(name)
    return {'retained_slots': kept, 'total_weight_slots': total, 'actual_nonzero_weights': nonzero,
            'remaining_slots_percent': 100*kept/total, 'actual_remaining_percent': 100*nonzero/total,
            'actual_sparsity': 1-nonzero/total, 'layers': layers}

@torch.no_grad()
def channel_bases(weight):
    w = weight.detach().double().cpu()
    return (torch.linalg.svd(w.flatten(1), full_matrices=False).U,
            torch.linalg.svd(w.permute(1,0,2,3).reshape(w.shape[1],-1), full_matrices=False).U)

@torch.no_grad()
def tucker_factors(weight, ranks, iterations=5, bases=None):
    w = weight.detach().double().cpu()
    ro, ri = ranks
    if w.ndim != 4 or not 1 <= ro <= w.shape[0] or not 1 <= ri <= w.shape[1] or iterations < 0:
        raise ValueError('Invalid channel ranks / iterations')
    u, v = channel_bases(w) if bases is None else bases
    u, v = u[:, :ro], v[:, :ri]
    for _ in range(iterations):
        a = torch.einsum('oihw,ib->obhw', w, v)
        u = torch.linalg.svd(a.flatten(1), full_matrices=False).U[:, :ro]
        a = torch.einsum('oa,oihw->aihw', u, w)
        v = torch.linalg.svd(a.permute(1,0,2,3).reshape(w.shape[1],-1), full_matrices=False).U[:, :ri]
    core = torch.einsum('oa,oihw,ib->abhw', u, w, v)
    return u, core, v

@torch.no_grad()
def tucker2_approx(weight, rank, iterations=5, bases=None):
    u, core, v = tucker_factors(weight, (rank,rank), iterations, bases)
    return torch.einsum('oa,abhw,ib->oihw', u, core, v).to(weight.dtype)

@torch.no_grad()
def factorized_conv(conv, ranks, iterations=5, bases=None):
    if conv.groups != 1:
        raise ValueError('Tucker-2 supports ungrouped convolution')
    ro, ri = ranks
    u, core, v = tucker_factors(conv.weight, ranks, iterations, bases)
    kwargs = dict(device=conv.weight.device, dtype=conv.weight.dtype)
    reduce = nn.Conv2d(conv.in_channels, ri, 1, bias=False, **kwargs)
    spatial = nn.Conv2d(ri, ro, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                        bias=False, padding_mode=conv.padding_mode, **kwargs)
    expand = nn.Conv2d(ro, conv.out_channels, 1, bias=conv.bias is not None, **kwargs)
    reduce.weight.copy_(v.T[:, :, None, None])
    spatial.weight.copy_(core)
    expand.weight.copy_(u[:, :, None, None])
    if conv.bias is not None:
        expand.bias.copy_(conv.bias)
    return nn.Sequential(reduce, spatial, expand)

@torch.no_grad()
def full_rank_check(conv):
    # Fixed private RNG, same input, preserve bias/stride/padding/dilation.
    g = torch.Generator().manual_seed(617)
    x = torch.randn(2, conv.in_channels, 9, 11, generator=g).to(conv.weight)
    full = factorized_conv(conv, (conv.out_channels,conv.in_channels), iterations=0)
    a, b = conv(x), full(x)
    error = float((a-b).abs().max())
    # FP32 summation order differs: absolute tolerance scales with output magnitude.
    scale = max(1., float(a.abs().max()))
    atol = 5e-6 * scale
    if not torch.allclose(a, b, atol=atol, rtol=2e-4):
        raise AssertionError(f'Full-rank Tucker output mismatch: {error}')
    return {'max_abs_error': error, 'atol': atol, 'rtol': 2e-4, 'output_scale': scale,
            'relative_frobenius_error': float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-30)), 'r_out': conv.out_channels, 'r_in': conv.in_channels}

@torch.no_grad()
def operation_cost(model):
    """Per-image multiply-accumulate count, Conv/Linear only; no latency claims."""
    macs, handles = [0], []
    def hook(layer, inp, out):
        if isinstance(layer, nn.Conv2d):
            macs[0] += out.numel()*layer.in_channels//layer.groups*layer.kernel_size[0]*layer.kernel_size[1]
        else:
            macs[0] += out.numel()*layer.in_features
    for layer in model.modules():
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            handles.append(layer.register_forward_hook(hook))
    training = model.training
    model.eval()
    p = next(model.parameters())
    model(torch.zeros(1,3,32,32, device=p.device, dtype=p.dtype))
    for handle in handles:
        handle.remove()
    model.train(training)
    return {'parameters': sum(p.numel() for p in model.parameters()), 'macs_per_image': macs[0]}


def analyze_checkpoint(directory, cfg, output, kind='all', layer='layer4.1.conv2',
                       keep=(.9,.85,.8,.75,.7,.65,.6,.55,.5), ranks=(60,50,40,30,25,20,15,10,5), iterations=5):
    directory, output = Path(directory), Path(output)
    cp_path = directory/'last.pt'
    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
    method, seed = cp['method'], cp['seed']
    trained_cfg = Config(**cp['config'])
    local = replace(trained_cfg, device=cfg.device, cpu_threads=cfg.cpu_threads, num_workers=cfg.num_workers,
                    data_dir=cfg.data_dir, test_limit=cfg.test_limit, precision='fp32')
    device = configure(local, method)
    model = create_model(method, seed).to(device)
    original = cp['model_state']
    model.load_state_dict(original)
    _, test = make_loaders(local, seed)
    output.mkdir(parents=True, exist_ok=True)
    base = epoch_pass(model, test, local, device)
    write_json(output/'metadata.json', {'method': method, 'seed': seed, 'epoch': cp['epoch'],
               'target_epochs': trained_cfg.epochs, 'test_images': len(test.dataset),
               'partial_training': cp['epoch'] < trained_cfg.epochs, 'training_config': cp['config'],
               'checkpoint': str(cp_path.resolve()), 'sha256': hashlib.sha256(cp_path.read_bytes()).hexdigest(),
               'baseline': base, 'environment': environment(local, device)})
    if kind in ('all','ranks'):
        write_csv(output/'kernel_ranks.csv', [{'method':method,'seed':seed,**r} for r in kernel_ranks(model)])
    if kind in ('all','pruning'):
        rows = []
        for fraction in keep:
            counts = magnitude_prune(model, original, fraction)
            metrics = epoch_pass(model, test, local, device)
            counts['layers'] = ';'.join(counts['layers'])
            rows.append({'method':method,'seed':seed,'requested_keep':fraction,**counts,**metrics})
            write_csv(output/'pruning.csv', rows)
    if kind in ('all','tucker'):
        model.load_state_dict(original)
        target = model.get_submodule(layer)
        if not isinstance(target,nn.Conv2d) or target.in_channels != 512 or target.out_channels != 512:
            raise ValueError('Figure6 requires a 512→512 convolution')
        check = full_rank_check(target)
        write_json(output/'tucker_full_rank_check.json',check)
        before = operation_cost(model)
        bases = channel_bases(target.weight)
        rows = []
        for rank in ranks:
            # Restore module type and every uncompressed parameter for each independent point.
            model.set_submodule(layer, target)
            model.load_state_dict(original)
            compressed = factorized_conv(target,(rank,rank),iterations,bases)
            model.set_submodule(layer,compressed)
            metrics = epoch_pass(model,test,local,device)
            after = operation_cost(model)
            rows.append({'method':method,'seed':seed,'layer':layer,'r_out':rank,'r_in':rank,
                         'iterations':iterations,'parameters_before':before['parameters'],
                         'parameters_after':after['parameters'],'macs_before':before['macs_per_image'],
                         'macs_after':after['macs_per_image'],**metrics})
            write_csv(output/'tucker.csv',rows)
        model.set_submodule(layer,target)
    return output
