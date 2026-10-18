import argparse
import json
from pathlib import Path
import platform

import torch
import torchvision
from torch import nn

from dataset import build_loaders
from engine import collect_mistakes, fit, run_epoch
from model import build_model, count_parameters
from plots import save_confusion_matrix, save_mistakes
from settings import PROFILES, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="teaching")
    parser.add_argument("--strategy", choices=["frozen", "partial"], required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    profile = PROFILES[args.profile]
    out = Path("outputs") / f"{args.profile}_{args.strategy}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"python={platform.python_version()} torch={torch.__version__} "
          f"torchvision={torchvision.__version__} device=cpu")
    loaders, _ = build_loaders("data", profile)
    print(f"samples train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")

    model = build_model(args.strategy).cpu()
    total, trainable = count_parameters(model)
    print(f"strategy={args.strategy} params={total:,} trainable={trainable:,}")
    history, seconds = fit(model, loaders, args.strategy, profile.epochs,
                           out / "best_model.pth")
    test_stats, cm = run_epoch(model, loaders["test"], nn.CrossEntropyLoss())
    mistakes = collect_mistakes(model, loaders["test"])
    save_confusion_matrix(cm, out / "confusion_matrix.png")
    save_mistakes(mistakes, out / "mistakes.png")

    result = {
        "profile": args.profile, "strategy": args.strategy,
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "device": "cpu",
        "samples": {key: len(loader.dataset) for key, loader in loaders.items()},
        "epochs": profile.epochs, "image_size": profile.image_size,
        "total_parameters": total, "trainable_parameters": trainable,
        "train_seconds": seconds, "test": test_stats, "confusion_matrix": cm.tolist(),
    }
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"test_acc={test_stats['accuracy']:.3f} macro_f1={test_stats['macro_f1']:.3f} "
          f"train_seconds={seconds:.1f}")
    print(f"artifacts={out.resolve()}")


if __name__ == "__main__":
    main()

