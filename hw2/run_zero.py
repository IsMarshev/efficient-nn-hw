"""CLI for the executable definition cells in zero-init.ipynb (no training on import)."""
import argparse
import contextlib
import io
import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path


def load_notebook():
    path = Path(__file__).with_name("zero-init.ipynb")
    notebook = json.loads(path.read_text())
    module = types.ModuleType("zero_notebook")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    for index, cell in enumerate(notebook["cells"]):
        if "zero-definitions" in cell.get("metadata", {}).get("tags", []):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile("".join(cell["source"]), f"{path}:cell-{index}", "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("h100_2h", "paper_cifar", "cifar_both", "smoke"), default="h100_2h")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/zero-init")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--budget-minutes", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--no-auto-fit", action="store_true")
    parser.add_argument("--plot-only", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    module = load_notebook()
    if args.plot_only:
        module.plot_results(args.plot_only)
        return
    cfg = replace(module.PRESETS[args.preset], device=args.device, data_dir=args.data_dir, output_dir=args.output_dir)
    for name in ("budget_minutes", "epochs", "num_workers", "precision"):
        value = getattr(args, name)
        if value is not None:
            cfg = replace(cfg, **{name: value})
    if args.epochs is not None:
        cfg = replace(cfg, min_epochs=min(cfg.min_epochs, cfg.epochs))
    if args.no_auto_fit:
        cfg = replace(cfg, auto_fit_budget=False)
    root, summaries = module.run_experiments(cfg, resume=args.resume)
    if list(root.glob("*/history.json")):
        module.plot_results(root)
    print("RESULTS:", root.resolve(), flush=True)
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "complete":
        print("INCOMPLETE: resume with --resume", root, flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
