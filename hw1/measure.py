"""Measure the 132-point CNN grid on one NVIDIA GPU.

Run ``python measure.py --help``. The script resumes an existing CSV by default.
Energy is measured for a continuous burst and divided by the number of passes.
NVML's hardware energy counter is preferred; sampled power is the fallback.
"""

import argparse
import csv
import gc
import json
import math
import random
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from equations import flops, memory
from models import make_model


BASE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FIELDS = ("image_size", "batch", "latency_s", "memory_bytes", "energy_j",
          "flops_profiler", "flops_predicted", "memory_predicted_bytes",
          "status", "is_validation", "energy_method")


def measurement_grid(seed):
    rng = random.Random(seed)
    sizes = sorted(set(BASE_SIZES) | set(rng.sample(
        [s for s in range(32, 513, 16) if s not in BASE_SIZES], 4)))
    batches = sorted(set(BASE_BATCHES) | set(rng.sample(
        [b for b in range(1, 257) if b & (b - 1)], 3)))
    return [(s, b, int(s not in BASE_SIZES or b not in BASE_BATCHES))
            for s in sizes for b in batches]


class GPUEnergyMeter:
    def __init__(self, device_index):
        self.nvml = None
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:
            print(f"NVML unavailable ({exc}); energy will be NaN")

    def _energy_counter(self):
        try:
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0
        except Exception:
            return None

    def _power(self):
        try:
            return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return None

    def measure_burst(self, run, interval):
        if self.handle is None:
            run()
            return float("nan"), "unavailable"

        samples = []
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                power = self._power()
                if power is not None:
                    samples.append((time.perf_counter(), power))
                stop.wait(interval)

        thread = threading.Thread(target=sample, daemon=True)
        before = self._energy_counter()
        t0 = time.perf_counter()
        p0 = self._power()
        thread.start()
        try:
            run()
        finally:
            t1 = time.perf_counter()
            stop.set()
            thread.join(timeout=1)
        p1 = self._power()
        after = self._energy_counter()
        if before is not None and after is not None and after >= before:
            return after - before, "nvml_counter"
        if p0 is None or p1 is None:
            return float("nan"), "unavailable"
        points = [(t0, p0)] + [(t, p) for t, p in samples if t0 < t < t1] + [(t1, p1)]
        joules = sum((tb - ta) * (pa + pb) / 2
                     for (ta, pa), (tb, pb) in zip(points, points[1:]))
        return joules, "nvml_power_samples"

    def close(self):
        if self.nvml is not None:
            self.nvml.nvmlShutdown()


def profiler_flops(model, x):
    """Profiler conv/linear FLOPs plus GAP and bias FLOPs it omits."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True) as prof:
        with torch.inference_mode():
            y = model(x)
            torch.cuda.synchronize()
            del y
    counted = sum(event.flops or 0 for event in prof.key_averages())
    b, _, s, _ = x.shape
    expected_counted = flops(s, b) - b * (2 * s * s + 356)
    if abs(counted - expected_counted) > max(1, 1e-6 * expected_counted):
        print(f"Profiler FLOPs incomplete at S={s}, B={b}: "
              f"{counted} vs expected {expected_counted}")
        return float("nan")
    return float(counted + b * (2 * s * s + 356))


def measure_one(model, meter, s, b, args):
    gc.collect()
    torch.cuda.empty_cache()
    x = torch.randn((b, 3, s, s), device="cuda", dtype=torch.float32)
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                y = model(x)
                del y
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            y = model(x)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            del y

            events = [(torch.cuda.Event(enable_timing=True),
                       torch.cuda.Event(enable_timing=True))
                      for _ in range(args.timed_repeats)]
            for start, end in events:
                start.record()
                y = model(x)
                end.record()
                del y
            torch.cuda.synchronize()
            latency_s = statistics.median(start.elapsed_time(end) / 1000.0
                                          for start, end in events)

            count = min(args.max_energy_repeats,
                        max(32, math.ceil(args.energy_min_seconds / max(latency_s, 1e-5))))

            def burst():
                for _ in range(count):
                    y = model(x)
                    del y
                torch.cuda.synchronize()

            total_joules, method = meter.measure_burst(burst, args.power_interval)
            energy_j = total_joules / count

        try:
            profiled = profiler_flops(model, x) if not args.skip_profiler else float("nan")
        except Exception as exc:
            print(f"Profiler failed at S={s}, B={b}: {exc}")
            profiled = float("nan")
        return latency_s, peak, energy_j, profiled, method
    finally:
        del x
        gc.collect()
        torch.cuda.empty_cache()


def write_environment(output_dir, grid, seed):
    import matplotlib
    import scipy

    props = torch.cuda.get_device_properties(0)
    details = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": seed,
        "image_sizes": sorted({s for s, _, _ in grid}),
        "batch_sizes": sorted({b for _, b, _ in grid}),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    (output_dir / "environment.json").write_text(json.dumps(details, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed-repeats", type=int, default=21)
    parser.add_argument("--energy-min-seconds", type=float, default=1.0)
    parser.add_argument("--max-energy-repeats", type=int, default=5000)
    parser.add_argument("--power-interval", type=float, default=0.02)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Run on Colab/Kaggle or an NVIDIA GPU.")
    if args.timed_repeats < 1 or args.max_energy_repeats < 1:
        parser.error("repeat counts must be positive")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    model = make_model().cuda().eval()
    grid = measurement_grid(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "measurements.csv"
    existing = {}
    if csv_path.exists() and not args.overwrite:
        with csv_path.open(newline="") as file:
            for row in csv.DictReader(file):
                existing[(int(row["image_size"]), int(row["batch"]))] = row
        expected = {(s, b): v for s, b, v in grid}
        for key, row in existing.items():
            if key not in expected or int(row["is_validation"]) != expected[key]:
                raise SystemExit("Existing CSV uses a different grid/seed; use --overwrite")
    write_environment(args.output, grid, args.seed)
    meter = GPUEnergyMeter(0)
    try:
        with csv_path.open("w" if args.overwrite or not csv_path.exists() else "a",
                           newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if args.overwrite or not existing:
                writer.writeheader()
            for i, (s, b, is_validation) in enumerate(grid, 1):
                if (s, b) in existing:
                    continue
                row = dict(image_size=s, batch=b, flops_predicted=flops(s, b),
                           memory_predicted_bytes=memory(s, b),
                           is_validation=is_validation)
                try:
                    latency_s, peak, energy_j, profiled, method = measure_one(
                        model, meter, s, b, args)
                    row.update(latency_s=latency_s, memory_bytes=peak,
                               energy_j=energy_j, flops_profiler=profiled,
                               energy_method=method, status="OK")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    row.update(latency_s="", memory_bytes="", energy_j="",
                               flops_profiler="", energy_method="", status="OOM")
                writer.writerow(row)
                file.flush()
                print(f"{i:3d}/{len(grid)} S={s:3d} B={b:3d} {row['status']}", flush=True)
    finally:
        meter.close()


if __name__ == "__main__":
    main()
