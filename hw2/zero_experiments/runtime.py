import os
import csv
import json
import random
import platform
import subprocess
import sys
import resource
import hashlib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torchvision


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.with_suffix('.csv.tmp').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix('.csv.tmp').replace(path)

def configure(cfg, method=None):
    strict = cfg.determinism == 'strict' or method == 'constnet'
    if method == 'constnet_star':
        if not cfg.device.startswith('cuda') or not torch.cuda.is_available():
            raise ValueError('ConstNet* requires nondeterministic CUDA; CPU/MPS is not equivalent')
        if cfg.determinism != 'original':
            raise ValueError('ConstNet* requires --determinism original')
        strict = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = not strict
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    if cfg.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires supported CUDA')
    return device

def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()

def autocast(cfg, device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if cfg.precision == 'bf16' else nullcontext()

def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)

def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}

def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if state.get('mps') is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])

def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(cpu_tree(v) for v in x)
    return x

def save_checkpoint(path, model, optimizer, scheduler, cfg, epoch, history, rank_rows, best, method, seed):
    path = Path(path)
    tmp = path.with_suffix('.pt.tmp')
    torch.save({'model_state': cpu_tree(model.state_dict()), 'optimizer_state': cpu_tree(optimizer.state_dict()),
                'scheduler_state': scheduler.state_dict(), 'rng_state': rng_state(), 'config': asdict(cfg),
                'epoch': epoch, 'history': history, 'rank_rows': rank_rows, 'best': best,
                'method': method, 'seed': seed}, tmp)
    tmp.replace(path)

def environment(cfg, device):
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'config': asdict(cfg), 'python': sys.version, 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'numpy': np.__version__, 'platform': platform.platform(),
            'hardware': platform.processor(), 'device': str(device),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine(),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'amp': cfg.precision, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short')}

def memory(device):
    # macOS ru_maxrss bytes; Linux KiB. Process high-water mark, not isolated GPU usage.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'process_peak_rss_mb': rss / (1024**2 if sys.platform == 'darwin' else 1024),
            'cuda_peak_allocated_mb': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None}
