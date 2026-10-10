"""Обучение MLP на MNIST; результаты сохраняются для последующего анализа."""
import argparse
from pathlib import Path
import torch
from zero_experiments.mnist import MLP, train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Воспроизведение MNIST-эксперимента ZerO")
    parser.add_argument("--method", choices=["ZerO", "Partial_Identity", "Random"], default="ZerO")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--root", type=Path, default=Path("runs/mnist-new"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = MLP(n_h=args.width, init=args.method)
    train_model(model, file_dir=args.root, epochs=args.epochs, seed=args.seed)
