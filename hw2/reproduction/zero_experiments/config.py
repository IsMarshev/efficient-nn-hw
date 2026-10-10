from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Config:
    methods: tuple = ('zero', 'kaiming', 'xavier')
    seeds: tuple = tuple(range(10))
    epochs: int = 200
    batch_size: int = 128
    lr: float = .1
    momentum: float = .9
    weight_decay: float = .0001
    warmup_epochs: int = 10
    warmup_start_lr: float = .01
    scheduler: str = 'linear_warmup_cosine'
    device: str = 'cpu'
    determinism: str = 'strict'
    precision: str = 'fp32'
    num_workers: int = 0
    cpu_threads: int = 4
    data_dir: str = 'data'
    output_dir: str = 'runs/reproduction'
    train_limit: int | None = None
    test_limit: int | None = None
    save_rank_weights: bool = True
    grad_steps: int = 400
    checkpoint_selection: str = 'last_epoch'
    protocol: str = 'cifar_reconstruction_v1'
    paper_version: str = '2110.12661v3'
    official_commit: str = 'cf2bf5d7d967958b3bb232755b288d97239896a8'

    def validate(self):
        from .models import METHODS
        if not self.methods or any(m not in METHODS for m in self.methods):
            raise ValueError('Invalid methods')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(s < 0 for s in self.seeds):
            raise ValueError('Seeds must be distinct nonnegative integers')
        if self.epochs < 1 or not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError('Invalid epochs/warmup')
        if self.batch_size < 1 or self.grad_steps < 1 or self.num_workers < 0:
            raise ValueError('Invalid batch size / steps / workers')
        if self.lr <= 0 or not 0 <= self.warmup_start_lr <= self.lr:
            raise ValueError('Invalid learning rate')
        if self.scheduler != 'linear_warmup_cosine' or self.checkpoint_selection != 'last_epoch':
            raise ValueError('Unsupported scheduler/checkpoint selection')
        if self.determinism not in ('strict', 'original') or self.precision not in ('fp32', 'bf16'):
            raise ValueError('Invalid runtime')
        for limit in (self.train_limit, self.test_limit):
            if limit is not None and limit < 1:
                raise ValueError('Data limits must be positive')
        if self.determinism == 'strict' and self.precision != 'fp32':
            raise ValueError('Strict protocol requires FP32')
        return self

    @classmethod
    def load(cls, path):
        cfg = cls(**json.loads(Path(path).read_text()))
        cfg.methods, cfg.seeds = tuple(cfg.methods), tuple(cfg.seeds)
        return cfg.validate()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))
