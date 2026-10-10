import time
import torch

LAYERS = ('layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1')
UNFOLDING = '[Cout,Cin*kH*kW]; full kernel; no identity subtraction'

@torch.no_grad()
def stable_rank(weight):
    w = weight.detach().double().cpu().flatten(1)
    if not torch.isfinite(w).all():
        raise FloatingPointError('Nonfinite kernel')
    if not torch.count_nonzero(w):
        return 0.0  # Convention for zero kernel, whose ratio is undefined.
    s = torch.linalg.svdvals(w)
    return float(s.square().sum() / s[0].square())

@torch.no_grad()
def snapshot(model, method, seed, epoch, directory, save_weights=True):
    started = time.perf_counter()
    weights = {name: model.get_submodule(name).weight.detach().cpu().clone() for name in LAYERS}
    rows = [{'method': method, 'seed': seed, 'epoch': epoch, 'layer': name,
             'stable_rank': stable_rank(w), 'unfolding': UNFOLDING} for name, w in weights.items()]
    if save_weights:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch, 'method': method, 'seed': seed, 'weights': weights,
                    'unfolding': UNFOLDING}, directory / f'epoch-{epoch:03d}.pt')
    return rows, time.perf_counter() - started

@torch.no_grad()
def kernel_ranks(model):
    rows = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) and '.shortcut.' not in name:
            rows.append({'index': len(rows), 'layer': name, 'stable_rank': stable_rank(layer.weight),
                         'unfolding': UNFOLDING})
    return rows
