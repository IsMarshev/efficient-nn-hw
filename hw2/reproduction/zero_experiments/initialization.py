import torch

@torch.no_grad()
def hadamard_matrix(size, *, device=None, dtype=torch.float32):
    if size < 1 or size & (size - 1):
        raise ValueError("Размер Hadamard должен быть степенью двойки")
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), dim=1),
                       torch.cat((h, -h), dim=1)), dim=0)
    return h


@torch.no_grad()
def structured_init_(weight, use_hadamard=True):
    if weight.ndim not in (2, 4):
        raise ValueError("Ожидается Linear [out, in] или Conv2d [out, in, kH, kW]")
    if weight.ndim == 4 and any(k % 2 == 0 for k in weight.shape[2:]):
        raise ValueError("Для центральной инициализации нужны нечётные размеры ядра")
    rows, cols = weight.shape[:2]
    if use_hadamard and rows > cols:
        size = 1 << (rows - 1).bit_length()
        matrix = hadamard_matrix(size, device=weight.device, dtype=weight.dtype)
        # Нормировка Algorithm 1 статьи: sqrt(2/N).
        matrix = matrix[:rows, :cols] * (2.0 / size) ** 0.5
    else:
        matrix = torch.eye(rows, cols, device=weight.device, dtype=weight.dtype)
    weight.zero_()
    if weight.ndim == 2:
        weight.copy_(matrix)
    else:
        weight[:, :, weight.shape[2] // 2, weight.shape[3] // 2].copy_(matrix)
    return weight


def zero_init_(weight):
    return structured_init_(weight, use_hadamard=True)


def identity_init_(weight):
    # Ablation: убираем Hadamard, сохраняя остальную схему ZerO.
    return structured_init_(weight, use_hadamard=False)