import math
import time
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
from .models import create_model
from .data import make_loaders, set_epoch, batch_hash, domain_seed
from .ranks import snapshot
from .runtime import (configure, synchronize, autocast, seed_process, write_json, write_csv,
                      save_checkpoint, restore_rng, environment, memory)


def learning_rate_at(step, total_steps, warmup_steps, peak_lr, warmup_start_lr=.01):
    if step < warmup_steps:
        return warmup_start_lr + (peak_lr-warmup_start_lr)*step/max(1, warmup_steps-1)
    progress = min(1., (step-warmup_steps)/max(1, total_steps-warmup_steps-1))
    return peak_lr * .5 * (1 + math.cos(math.pi*progress))

class Schedule:
    def __init__(self, optimizer, cfg, batches):
        self.optimizer, self.cfg, self.batches, self.step_index = optimizer, cfg, batches, 0

    def apply(self):
        lr = learning_rate_at(self.step_index, self.cfg.epochs*self.batches,
                              self.cfg.warmup_epochs*self.batches, self.cfg.lr, self.cfg.warmup_start_lr)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def state_dict(self):
        return {'step_index': self.step_index, 'batches': self.batches, 'epochs': self.cfg.epochs,
                'warmup_epochs': self.cfg.warmup_epochs, 'lr': self.cfg.lr, 'warmup_start_lr': self.cfg.warmup_start_lr}

    def load_state_dict(self, state):
        current = self.state_dict()
        if any(current[k] != state[k] for k in current if k != 'step_index'):
            raise ValueError('Cannot change saved schedule on resume')
        self.step_index = state['step_index']


def optimizer_for(model, cfg):
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def epoch_pass(model, loader, cfg, device, optimizer=None, scheduler=None, grad_rows=None,
               method=None, seed=0, mode='warmup', max_steps=None):
    training = optimizer is not None
    model.train(training)
    synchronize(device)
    started = time.perf_counter()
    totals = torch.zeros(3, device=device)
    batches = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if max_steps is not None and scheduler.step_index >= max_steps:
                break
            step = scheduler.step_index if training else None
            fingerprint = batch_hash(images, labels) if training and grad_rows is not None and step < cfg.grad_steps else None
            images, labels = images.to(device), labels.to(device)
            if training:
                lr = scheduler.apply()
                optimizer.zero_grad(set_to_none=True)
            with autocast(cfg, device):
                output = model(images)
                loss = nn.functional.cross_entropy(output, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss, method={method}, step={step}')
            if training:
                loss.backward()
                # BF16 uses no GradScaler. No clipping. Capture before optimizer.step.
                if grad_rows is not None and step < cfg.grad_steps:
                    norm = float(torch.linalg.vector_norm(model.conv1.weight.grad.detach().float()))
                    if not math.isfinite(norm):
                        raise FloatingPointError(f'Nonfinite stem gradient at step {step}')
                    grad_rows.append({'method': method, 'seed': seed, 'mode': mode, 'step': step,
                                      'layer': 'conv1.weight', 'norm_type': 'L2/Frobenius', 'gradient_norm': norm,
                                      'lr': lr, 'loss': float(loss.detach()), 'batch_sha256': fingerprint})
                optimizer.step()
                scheduler.step_index += 1
            n = labels.numel()
            totals[0] += loss.detach().float()*n
            totals[1] += (output.argmax(1)==labels).sum()
            totals[2] += n
            batches += 1
    synchronize(device)
    loss_sum, correct, count = totals.tolist()
    if count == 0:
        raise ValueError('Empty epoch')
    return {'loss': loss_sum/count, 'accuracy': 100*correct/count, 'error': 100*(1-correct/count),
            'images': int(count), 'batches': batches, 'seconds': time.perf_counter()-started}


def train_one(cfg, method, seed, directory, resume=False, loaders=None, stop_after=None):
    cfg.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'last.pt').exists() and not resume:
        raise FileExistsError(f'{directory}: use --resume')
    device = configure(cfg, method)
    seed_process(domain_seed(seed, 'process'))
    train, test = loaders if loaders is not None else make_loaders(cfg, seed)
    model = create_model(method, domain_seed(seed, 'initialization')).to(device)
    optimizer = optimizer_for(model, cfg)
    scheduler = Schedule(optimizer, cfg, len(train))
    history, rank_rows, best, first_epoch = [], [], {'accuracy': -1., 'epoch': None}, 0
    if resume:
        cp = torch.load(directory/'last.pt', map_location='cpu', weights_only=True)
        saved = cp['config']
        current = asdict(cfg)
        allowed = ('device', 'data_dir', 'output_dir', 'cpu_threads')
        if any(current[k] != saved[k] for k in current if k not in allowed) or cp['method'] != method or cp['seed'] != seed:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(cp['model_state'])
        optimizer.load_state_dict(cp['optimizer_state'])
        scheduler.load_state_dict(cp['scheduler_state'])
        restore_rng(cp['rng_state'])
        history, rank_rows, best, first_epoch = cp['history'], cp['rank_rows'], cp['best'], cp['epoch']
        del cp
    write_json(directory/'config.json', {**asdict(cfg), 'method': method, 'seed': seed, 'architecture': 'resnet18'})
    write_json(directory/'environment.json', environment(cfg, device))
    grad_rows = []
    grad_file = directory/'gradients.json'
    if resume and grad_file.exists():
        import json
        grad_rows = [r for r in json.loads(grad_file.read_text()) if r['step'] < scheduler.step_index]
    if not history:
        initial = epoch_pass(model, test, cfg, device)
        rows, rank_seconds = snapshot(model, method, seed, 0, directory/'rank_weights', cfg.save_rank_weights)
        rank_rows.extend(rows)
        history.append({'epoch': 0, 'lr': 0., 'train_loss': None, 'train_accuracy': None, 'train_error': None,
                        'test_loss': initial['loss'], 'test_accuracy': initial['accuracy'], 'test_error': initial['error'],
                        'train_seconds': 0., 'test_seconds': initial['seconds'], 'rank_seconds': rank_seconds})
        save_checkpoint(directory/'last.pt', model, optimizer, scheduler, cfg, 0, history, rank_rows, best, method, seed)
    for epoch in range(first_epoch, cfg.epochs):
        if stop_after is not None and epoch >= stop_after:
            break
        set_epoch(train, seed, epoch)
        training = epoch_pass(model, train, cfg, device, optimizer, scheduler, grad_rows, method, seed,
                              'warmup' if cfg.warmup_epochs else 'no_warmup')
        evaluation = epoch_pass(model, test, cfg, device)
        rows, rank_seconds = snapshot(model, method, seed, epoch+1, directory/'rank_weights', cfg.save_rank_weights)
        rank_rows.extend(rows)
        history.append({'epoch': epoch+1, 'lr': optimizer.param_groups[0]['lr'],
                        'train_loss': training['loss'], 'train_accuracy': training['accuracy'], 'train_error': training['error'],
                        'test_loss': evaluation['loss'], 'test_accuracy': evaluation['accuracy'], 'test_error': evaluation['error'],
                        'train_seconds': training['seconds'], 'test_seconds': evaluation['seconds'], 'rank_seconds': rank_seconds})
        improved = evaluation['accuracy'] > best['accuracy']
        if improved:
            best = {'accuracy': evaluation['accuracy'], 'epoch': epoch+1}
        if improved:
            save_checkpoint(directory/'best_observed_test.pt', model, optimizer, scheduler, cfg, epoch+1,
                            history, rank_rows, best, method, seed)
        save_checkpoint(directory/'last.pt', model, optimizer, scheduler, cfg, epoch+1, history, rank_rows, best, method, seed)
        write_json(grad_file, grad_rows)
        write_csv(directory/'gradients.csv', grad_rows)
        write_json(directory/'history.json', history)
        write_csv(directory/'history.csv', history)
        write_csv(directory/'ranks.csv', rank_rows)
        print(f'{method}/seed{seed} {epoch+1}/{cfg.epochs}: test={evaluation["accuracy"]:.2f}% train={training["seconds"]:.2f}s', flush=True)
    complete = history[-1]['epoch'] == cfg.epochs
    summary = {'method': method, 'seed': seed, 'architecture': 'resnet18', 'status': 'complete' if complete else 'partial',
               'epochs': history[-1]['epoch'], 'target_epochs': cfg.epochs, 'test_accuracy': history[-1]['test_accuracy'],
               'test_error': history[-1]['test_error'], 'test_loss': history[-1]['test_loss'],
               'best_observed_test': best, 'checkpoint_selection': 'last_epoch',
               'train_images': len(train.dataset), 'test_images': len(test.dataset),
               'smoke': cfg.train_limit is not None or cfg.test_limit is not None or cfg.epochs < 200,
               'train_seconds': sum(r['train_seconds'] for r in history),
               'test_seconds': sum(r['test_seconds'] for r in history),
               'rank_seconds': sum(r['rank_seconds'] for r in history), **memory(device)}
    write_json(directory/'summary.json', summary)
    write_json(directory/'history.json', history)
    write_csv(directory/'history.csv', history)
    write_csv(directory/'ranks.csv', rank_rows)
    return summary


def run_suite(cfg, root, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'run_config.json').exists():
        from .config import Config
        old = Config.load(root/'run_config.json')
        if not resume:
            raise FileExistsError('Use a new output root or --resume')
        if any(asdict(old)[k] != asdict(cfg)[k] for k in asdict(cfg) if k not in ('device','data_dir','output_dir','cpu_threads')):
            raise ValueError('Suite resume configuration mismatch')
    cfg.save(root/'run_config.json')
    summaries = []
    for seed in cfg.seeds:
        for method in cfg.methods:
            directory = root/f'resnet18-{method}-seed{seed}'
            summaries.append(train_one(cfg, method, seed, directory, resume=resume and (directory/'last.pt').exists()))
            write_csv(root/'summary.csv', summaries)
    write_json(root/'status.json', {'status': 'complete', 'completed_runs': len(summaries)})
    return summaries


def gradient_suite(cfg, root, steps=None):
    """Six short runs, preserving the full-dataset warmup horizon."""
    from dataclasses import replace
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    steps = steps or cfg.grad_steps
    if cfg.precision != 'fp32':
        raise ValueError('Figure 8 uses FP32; no AMP scaler ambiguity')
    all_rows = []
    for seed in cfg.seeds:
        hashes = None
        for mode in ('warmup', 'no_warmup'):
            local = replace(cfg, warmup_epochs=cfg.warmup_epochs if mode == 'warmup' else 0, grad_steps=steps)
            for method in ('zero', 'kaiming', 'xavier'):
                device = configure(local, method)
                seed_process(domain_seed(seed, 'process'))
                loader, _ = make_loaders(local, seed)
                model = create_model(method, domain_seed(seed, 'initialization')).to(device)
                optimizer = optimizer_for(model, local)
                scheduler = Schedule(optimizer, local, len(loader))
                rows = []
                started = time.perf_counter()
                try:
                    for epoch in range(math.ceil(steps/len(loader))):
                        set_epoch(loader, seed, epoch)
                        epoch_pass(model, loader, local, device, optimizer, scheduler, rows, method, seed, mode, steps)
                    status = 'complete'
                except FloatingPointError as e:
                    status = 'diverged'
                    write_json(root/f'{method}-{mode}-seed{seed}-failure.json', {'status': status, 'error': str(e), 'step': scheduler.step_index})
                actual = [r['batch_sha256'] for r in rows]
                if hashes is None:
                    hashes = actual
                if actual != hashes[:len(actual)]:
                    raise AssertionError('Figure 8 batches/augmentation differ')
                all_rows.extend(rows)
                write_json(root/'gradients.json', all_rows)
                write_csv(root/'gradients.csv', all_rows)
                write_json(root/f'{method}-{mode}-seed{seed}-metadata.json',
                           {**environment(local, device), 'status': status, 'recorded_steps': len(rows),
                            'requested_steps': steps, 'seconds': time.perf_counter()-started, **memory(device)})
                print(f'Figure8 {method}/{mode}: {len(rows)} steps {status}', flush=True)
    cfg.save(root/'run_config.json')
    return all_rows


def benchmark(cfg, root, batches=3):
    """Isolated throughput probe. No benchmark model is used as an experiment checkpoint."""
    from itertools import islice
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    device = configure(cfg)
    seed = cfg.seeds[0]
    train,test = make_loaders(cfg,seed)
    set_epoch(train,seed,0)
    model = create_model(cfg.methods[0],domain_seed(seed,'initialization')).to(device)
    opt = optimizer_for(model,cfg)
    schedule = Schedule(opt,cfg,len(train))
    probe = list(islice(train,batches))
    evaluation = list(islice(test,batches))
    epoch_pass(model,probe[:1],cfg,device,opt,schedule)  # warmup excluded
    measured = epoch_pass(model,probe,cfg,device,opt,schedule)
    tested = epoch_pass(model,evaluation,cfg,device)
    rank_start = time.perf_counter()
    from .ranks import stable_rank, LAYERS
    for name in LAYERS:
        stable_rank(model.get_submodule(name).weight)
    rank_seconds = time.perf_counter()-rank_start
    train_epoch = measured['seconds']/measured['images']*len(train.dataset)
    test_epoch = tested['seconds']/tested['images']*len(test.dataset)
    estimated_epoch = train_epoch+test_epoch+rank_seconds
    result = {'environment':environment(cfg,device),'measured_batches':batches,
              'training_probe':measured,'evaluation_probe':tested,'rank_seconds':rank_seconds,
              'estimated_epoch_seconds_excluding_checkpoint_io':estimated_epoch,
              'estimated_main_30_runs_hours':estimated_epoch*cfg.epochs*30/3600,
              'estimated_figure8_six_runs_seconds':measured['seconds']/measured['batches']*400*6,
              'estimated_rank_snapshots_bytes_per_run':sum(model.get_submodule(n).weight.numel()*4 for n in LAYERS)*(cfg.epochs+1),
              'warning':'Extrapolation of a short CPU/GPU probe, not a completed training result; excludes IO and hardware load changes',
              **memory(device)}
    write_json(root/'benchmark.json',result)
    print(result,flush=True)
    return result
