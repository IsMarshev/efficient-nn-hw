"""Analytical FP32 forward-pass costs for :mod:`models`.

All four public functions broadcast NumPy inputs. A scalar input pair produces a
Python float. Image sizes must be positive multiples of 16 and batches positive.
"""

import numpy as np


PARAMETERS = 1_040_324
PARAMETER_BYTES = 4 * PARAMETERS
KERNEL_LAUNCHES = 17  # six convs, seven ReLUs, pool, GAP, two linears


def _inputs(image_size, batch):
    s, b = np.broadcast_arrays(np.asarray(image_size), np.asarray(batch))
    if np.any(~np.isfinite(s)) or np.any(~np.isfinite(b)):
        raise ValueError("image_size and batch must be finite")
    if np.any(s < 16) or np.any(s % 16 != 0) or np.any(b < 1) or np.any(b % 1 != 0):
        raise ValueError("image_size must be a positive multiple of 16; batch must be positive")
    return s.astype(np.float64), b.astype(np.float64)


def _out(value):
    return float(value) if np.ndim(value) == 0 else value


def flops(image_size, batch):
    """FLOPs; one multiply-accumulate counts as two operations.

    Includes convolution, global mean (additions and divisions), both linear
    layers and their bias additions. ReLU and max-pooling are comparisons.
    """
    s, b = _inputs(image_size, batch)
    return _out(b * (17_714 * s**2 + 313_700))


def memory(image_size, batch):
    """Ideal peak live FP32 tensor bytes, including parameters and caller input.

    The peak occurs at max-pooling: the caller-held RGB input, first conv
    output, and pool output coexist (3 + 8 + 2) * B * S**2 elements.
    CUDA context, caching, and temporary cuDNN workspaces are not modeled.
    """
    s, b = _inputs(image_size, batch)
    return _out(PARAMETER_BYTES + 52 * b * s**2)


def bytes_moved(image_size, batch):
    """Lower-bound bytes for separate FP32 kernels; weights read once per pass.

    Counts one input read and output write per operator, including in-place
    ReLUs; excludes cache effects, repeated convolution reads, and workspaces.
    """
    s, b = _inputs(image_size, batch)
    return _out(4 * (PARAMETERS + b * (91 * s**2 + 2_148)))


def _operator_costs(s, b):
    """Return (FLOPs, bytes) for each launched operator, in execution order."""
    n = b * s**2
    z = np.zeros_like(n)

    def op(f, elements, weights=0):
        return np.asarray(f), 4 * (np.asarray(elements) + weights)

    costs = [
        op(2_352*n, 11*n, 4_704),       # conv7
        op(z, 16*n),                      # ReLU
        op(z, 10*n),                      # maxpool
        op(6_400*n, 6*n, 51_200),       # conv5
        op(z, 8*n),
        op(2_304*n, 6*n, 73_728),       # conv3
        op(z, 4*n),
        op(1_024*n, 6*n, 32_768),       # conv1
        op(z, 8*n),
        op(4_608*n, 5*n, 589_824),      # conv3
        op(z, 2*n),
        op(1_024*n, 3*n, 131_072),      # conv1
        op(z, 4*n),
        op(2*n, 2*n + 512*b),           # global average pool
        op(262_400*b, 768*b, 131_328), # 512->256, including bias additions
        op(z, 512*b),
        op(51_300*b, 356*b, 25_700),   # 256->100, including bias additions
    ]
    return costs


def latency(image_size, batch, theta):
    """Predicted forward latency in seconds.

    theta: {launch_seconds, effective_flops_per_second, effective_bytes_per_second}.
    Each sequential kernel takes the maximum of launch, arithmetic and traffic
    time. Effective rates are fitted on the training configurations.
    """
    s, b = _inputs(image_size, batch)
    launch = float(theta["launch_seconds"])
    flop_rate = float(theta["effective_flops_per_second"])
    byte_rate = float(theta["effective_bytes_per_second"])
    if min(launch, flop_rate, byte_rate) <= 0:
        raise ValueError("latency parameters must be positive")
    total = np.zeros_like(s, dtype=np.float64)
    for f, moved in _operator_costs(s, b):
        total += np.maximum.reduce((np.broadcast_to(launch, s.shape),
                                    f / flop_rate, moved / byte_rate))
    return _out(total)


def energy(image_size, batch, theta_energy):
    """Predicted whole-GPU joules for one forward pass.

    E = idle_power_W * predicted_latency + joules_per_flop * FLOPs
        + joules_per_byte * lower_bound_bytes_moved.
    theta_energy contains the three coefficients and nested ``latency`` theta.
    Nonnegative effective coefficients are fitted from GPU energy measurements.
    """
    p = float(theta_energy["idle_power_watts"])
    ef = float(theta_energy["joules_per_flop"])
    eb = float(theta_energy["joules_per_byte"])
    if min(p, ef, eb) < 0:
        raise ValueError("energy parameters must be nonnegative")
    result = (p * latency(image_size, batch, theta_energy["latency"])
              + ef * flops(image_size, batch)
              + eb * bytes_moved(image_size, batch))
    return _out(result)
