"""Fit effective GPU parameters and plot predictions against measurements."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, energy, flops, latency, memory
from measure import measurement_grid


LATENCY_KEYS = ("launch_seconds", "effective_flops_per_second",
                "effective_bytes_per_second")


def read_rows(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No measurements in {path}")
    for row in rows:
        row["image_size"] = int(row["image_size"])
        row["batch"] = int(row["batch"])
        row["is_validation"] = bool(int(row["is_validation"]))
        for key in ("latency_s", "memory_bytes", "energy_j", "flops_profiler"):
            row[key] = float(row[key]) if row.get(key) else float("nan")
    return rows


def validate_results(rows, seed):
    """Reject partial or inconsistent measurements before calling them results."""
    expected = {(s, b): bool(validation)
                for s, b, validation in measurement_grid(seed)}
    actual = {(r["image_size"], r["batch"]): r for r in rows}
    if len(rows) != len(actual) or set(actual) != set(expected):
        missing = len(set(expected) - set(actual))
        extra = len(set(actual) - set(expected))
        raise ValueError(f"Incomplete grid: expected 132 unique points; "
                         f"missing={missing}, extra={extra}, duplicate={len(rows)-len(actual)}. "
                         "Finish measure.py before calibrating.")
    for key, row in actual.items():
        if row["is_validation"] != expected[key]:
            raise ValueError(f"Wrong validation flag at S={key[0]}, B={key[1]}")
        if row["status"] not in ("OK", "OOM"):
            raise ValueError(f"Unknown status at S={key[0]}, B={key[1]}")
        if row["status"] == "OK":
            for field in ("latency_s", "memory_bytes", "energy_j"):
                if not np.isfinite(row[field]) or row[field] <= 0:
                    raise ValueError(f"Missing {field} at S={key[0]}, B={key[1]}; "
                                     "whole-GPU energy requires working NVML.")
    for split in (False, True):
        if not any(r["status"] == "OK" and r["is_validation"] == split
                   and np.isfinite(r["flops_profiler"]) for r in rows):
            name = "validation" if split else "training"
            raise ValueError(f"No profiler FLOPs in {name} points; "
                             "rerun measurement without --skip-profiler.")


def arrays(rows, field):
    valid = [r for r in rows if r["status"] == "OK" and
             np.isfinite(r[field]) and r[field] > 0]
    return (np.array([r["image_size"] for r in valid], dtype=float),
            np.array([r["batch"] for r in valid], dtype=float),
            np.array([r[field] for r in valid], dtype=float), valid)


def fit_latency(rows):
    train = [r for r in rows if not r["is_validation"]]
    s, b, y, _ = arrays(train, "latency_s")
    if len(y) < 8:
        raise ValueError("At least eight successful training latency points are required")
    lower = np.log10([1e-8, 1e7, 1e5])
    upper = np.log10([1e-1, 1e15, 1e13])

    def residual(log_values):
        theta = dict(zip(LATENCY_KEYS, 10**log_values))
        return np.log(latency(s, b, theta) / y)

    starts = ([5e-6, 5e12, 2e11], [1e-5, 1e12, 1e11],
              [1e-4, 1e11, 1e10], [1e-6, 1e13, 1e12])
    fits = [least_squares(residual, np.log10(start), bounds=(lower, upper),
                          loss="soft_l1", f_scale=0.2, max_nfev=2000)
            for start in starts]
    best = min(fits, key=lambda fit: np.sum(residual(fit.x)**2))
    return dict(zip(LATENCY_KEYS, map(float, 10**best.x)))


def fit_energy(rows, theta_latency):
    train = [r for r in rows if not r["is_validation"]]
    s, b, y, _ = arrays(train, "energy_j")
    if len(y) < 3:
        return None
    design = np.column_stack((latency(s, b, theta_latency),
                              flops(s, b), bytes_moved(s, b)))
    scale = np.maximum(np.max(design, axis=0), 1e-30)
    coefficients, _ = nnls(design / scale, y)
    p, ef, eb = coefficients / scale
    return {"idle_power_watts": float(p), "joules_per_flop": float(ef),
            "joules_per_byte": float(eb), "latency": theta_latency}


def error_metrics(actual, predicted):
    if not len(actual):
        return {"count": 0}
    relative = np.abs(predicted - actual) / actual
    return {"count": len(actual),
            "median_absolute_percentage_error": float(np.median(relative) * 100),
            "mean_absolute_percentage_error": float(np.mean(relative) * 100),
            "median_signed_percentage_error": float(np.median((predicted / actual - 1) * 100))}


def metric_summary(rows, field, predict):
    answer = {}
    for label, subset in (("training", [r for r in rows if not r["is_validation"]]),
                          ("validation", [r for r in rows if r["is_validation"]])):
        s, b, y, _ = arrays(subset, field)
        answer[label] = error_metrics(y, predict(s, b))
    return answer


def largest_errors(rows, field, predict, count=3):
    candidates = [r for r in rows if r["is_validation"] and r["status"] == "OK"
                  and np.isfinite(r[field]) and r[field] > 0]
    ranked = []
    for r in candidates:
        predicted = float(predict(r["image_size"], r["batch"]))
        ranked.append({"image_size": r["image_size"], "batch": r["batch"],
                       "measured": r[field], "predicted": predicted,
                       "absolute_percentage_error":
                       100 * abs(predicted / r[field] - 1)})
    return sorted(ranked, key=lambda r: r["absolute_percentage_error"],
                  reverse=True)[:count]


def plot_metric(rows, field, predict, title, unit, factor, path):
    sizes = np.array(sorted({r["image_size"] for r in rows}))
    batches = np.array(sorted({r["batch"] for r in rows}))
    ss, bb = np.meshgrid(sizes, batches, indexing="ij")
    predicted = predict(ss, bb) * factor
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(ss, bb, predicted, cmap="viridis", alpha=0.55,
                    linewidth=0, antialiased=True)
    for is_validation, color, label in ((False, "tab:blue", "Measured: fit"),
                                        (True, "tab:orange", "Measured: validation")):
        subset = [r for r in rows if r["status"] == "OK" and
                  r["is_validation"] == is_validation and
                  np.isfinite(r[field])]
        if subset:
            ax.scatter([r["image_size"] for r in subset],
                       [r["batch"] for r in subset],
                       [r[field] * factor for r in subset],
                       color=color, s=18, depthshade=False, label=label)
    if field == "memory_bytes":
        oom = [r for r in rows if r["status"] == "OOM"]
        if oom:
            ax.scatter([r["image_size"] for r in oom],
                       [r["batch"] for r in oom],
                       [memory(r["image_size"], r["batch"]) * factor for r in oom],
                       color="tab:red", marker="x", s=30, label="OOM (predicted height)")
    ax.set(xlabel="Image side S (pixels)", ylabel="Batch B (images)",
           zlabel=unit, title=title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def update_readme(summary, environment, path):
    start = "<!-- AUTO_RESULTS_START -->"
    end = "<!-- AUTO_RESULTS_END -->"
    content = path.read_text()
    if start not in content or end not in content:
        return
    lines = ["### Measured results", "",
             f"GPU: {environment.get('gpu', 'unknown')}; PyTorch "
             f"{environment.get('torch', 'unknown')}; CUDA "
             f"{environment.get('torch_cuda', 'unknown')}; cuDNN "
             f"{environment.get('cudnn', 'unknown')}.", "",
             f"Configurations: {summary['total_configurations']}; "
             f"OOM: {summary['oom_configurations']}.", "",
             "| Quantity | Fit median absolute error | Validation median absolute error |",
             "|---|---:|---:|"]
    for name in ("flops", "memory", "latency", "energy"):
        if name not in summary:
            continue
        values = summary[name]
        def cell(split):
            v = values[split]
            return (f"{v['median_absolute_percentage_error']:.2f}% (n={v['count']})"
                    if v.get("count") else "unavailable")
        lines.append(f"| {name} | {cell('training')} | {cell('validation')} |")
    if "oom_vs_ideal_memory" in summary:
        oom = summary["oom_vs_ideal_memory"]
        lines.extend(["", f"OOM cases below the ideal-memory capacity threshold: "
                      f"{oom['oom_below_ideal_capacity']}."])
    if summary.get("largest_validation_errors"):
        lines.extend(["", "Largest validation deviations:", "",
                      "| Quantity | S | B | Measured | Predicted | Absolute error |",
                      "|---|---:|---:|---:|---:|---:|"])
        for name, unit in (("memory", "MiB"), ("latency", "ms"),
                           ("energy", "mJ")):
            worst = summary["largest_validation_errors"].get(name, [])
            if not worst:
                continue
            r = worst[0]
            factor = {"memory": 1 / 2**20, "latency": 1e3,
                      "energy": 1e3}[name]
            lines.append(f"| {name} ({unit}) | {r['image_size']} | {r['batch']} | "
                         f"{r['measured'] * factor:.3g} | "
                         f"{r['predicted'] * factor:.3g} | "
                         f"{r['absolute_percentage_error']:.1f}% |")
    lines.extend(["", "Figures: [FLOPs](results/figures/flops.png), "
                  "[memory](results/figures/memory.png), "
                  "[latency](results/figures/latency.png), "
                  "[energy](results/figures/energy.png)."])
    replacement = start + "\n" + "\n".join(lines) + "\n" + end
    left, rest = content.split(start, 1)
    _, right = rest.split(end, 1)
    path.write_text(left + replacement + right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    rows = read_rows(args.results / "measurements.csv")
    env_path = args.results / "environment.json"
    environment = json.loads(env_path.read_text()) if env_path.exists() else {}
    validate_results(rows, environment.get("seed", 2026))
    theta_latency = fit_latency(rows)
    theta_energy = fit_energy(rows, theta_latency)
    if theta_energy is None:
        raise ValueError("Not enough GPU energy measurements to fit theta_energy")
    theta = {"latency": theta_latency, "energy": theta_energy}
    (args.results / "theta.json").write_text(json.dumps(theta, indent=2) + "\n")

    summary = {
        "total_configurations": len(rows),
        "oom_configurations": sum(r["status"] == "OOM" for r in rows),
        "flops": metric_summary(rows, "flops_profiler", flops),
        "memory": metric_summary(rows, "memory_bytes", memory),
        "latency": metric_summary(rows, "latency_s",
                                  lambda s, b: latency(s, b, theta_latency)),
    }
    summary["energy"] = metric_summary(
        rows, "energy_j", lambda s, b: energy(s, b, theta_energy))
    summary["largest_validation_errors"] = {
        "memory": largest_errors(rows, "memory_bytes", memory),
        "latency": largest_errors(rows, "latency_s",
                                  lambda s, b: latency(s, b, theta_latency)),
        "energy": largest_errors(rows, "energy_j",
                                 lambda s, b: energy(s, b, theta_energy)),
    }
    if "gpu_total_memory_bytes" in environment:
        total_bytes = environment["gpu_total_memory_bytes"]
        summary["oom_vs_ideal_memory"] = {
            "predicted_over_physical_capacity": sum(
                memory(r["image_size"], r["batch"]) > total_bytes for r in rows),
            "oom_below_ideal_capacity": sum(
                r["status"] == "OOM" and
                memory(r["image_size"], r["batch"]) <= total_bytes for r in rows),
            "successful_above_ideal_capacity": sum(
                r["status"] == "OK" and
                memory(r["image_size"], r["batch"]) > total_bytes for r in rows),
        }
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if any(np.isfinite(r["flops_profiler"]) for r in rows):
        plot_metric(rows, "flops_profiler", flops, "FLOPs: profiler vs equation",
                    "FLOPs (GFLOPs)", 1e-9, figures / "flops.png")
    plot_metric(rows, "memory_bytes", memory, "Peak allocated memory",
                "Memory (MiB)", 1 / 2**20, figures / "memory.png")
    plot_metric(rows, "latency_s", lambda s, b: latency(s, b, theta_latency),
                "Forward latency", "Latency (ms)", 1e3, figures / "latency.png")
    plot_metric(rows, "energy_j", lambda s, b: energy(s, b, theta_energy),
                "Whole-GPU energy per pass", "Energy (mJ)", 1e3,
                figures / "energy.png")
    update_readme(summary, environment, Path(__file__).parent / "README.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
