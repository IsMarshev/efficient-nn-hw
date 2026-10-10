"""Independent, worker-invariant RNG for sampler and per-image augmentation."""
import hashlib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor
from pathlib import Path

MEAN = torch.tensor((.4914, .4822, .4465))[:, None, None]
STD = torch.tensor((.2470, .2435, .2616))[:, None, None]

def domain_seed(seed, domain, epoch=0, index=0):
    digest = hashlib.sha256(f'{seed}/{domain}/{epoch}/{index}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2**63 - 1)

class AugmentedCIFAR(Dataset):
    def __init__(self, root, train, seed, limit=None):
        self.base = CIFAR10(root, train=train, download=False)
        self.train, self.seed, self.epoch = train, seed, 0
        self.size = min(limit or len(self.base), len(self.base))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image, label = self.base[index]
        image = to_tensor(image)
        if self.train:
            g = torch.Generator().manual_seed(domain_seed(self.seed, 'augmentation', self.epoch, index))
            top, left = torch.randint(0, 9, (2,), generator=g).tolist()
            image = F.pad(image, (4, 4, 4, 4))[:, top:top+32, left:left+32]
            if torch.rand((), generator=g).item() < .5:
                image = image.flip(2)
        return (image - MEAN) / STD, label

def ensure_data(root):
    # torchvision verifies all extracted CIFAR MD5s before consuming pickle.
    from torchvision.datasets.utils import check_integrity, download_and_extract_archive
    root = Path(root)
    expected = CIFAR10.train_list + CIFAR10.test_list + [(CIFAR10.meta['filename'], CIFAR10.meta['md5'])]
    if not all(check_integrity(str(root / CIFAR10.base_folder / n), md5) for n, md5 in expected):
        download_and_extract_archive('https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/main/cifar-10-python.tar.gz',
                                    str(root), filename=CIFAR10.filename, md5=CIFAR10.tgz_md5)

def make_loaders(cfg, seed):
    ensure_data(cfg.data_dir)
    train = AugmentedCIFAR(cfg.data_dir, True, seed, cfg.train_limit)
    test = AugmentedCIFAR(cfg.data_dir, False, seed, cfg.test_limit)
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.device.startswith('cuda'))
    return (DataLoader(train, shuffle=True, generator=torch.Generator(), **common),
            DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(domain_seed(seed, 'eval')), **common))

def set_epoch(loader, seed, epoch):
    if hasattr(loader.dataset, 'epoch'):
        loader.dataset.epoch = epoch
    if loader.generator is not None:
        loader.generator.manual_seed(domain_seed(seed, 'sampler', epoch))

def batch_hash(images, labels):
    h = hashlib.sha256(images.contiguous().numpy().tobytes())
    h.update(labels.contiguous().numpy().tobytes())
    return h.hexdigest()
