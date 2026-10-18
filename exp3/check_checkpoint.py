import argparse
from pathlib import Path

import torch

from dataset import build_loaders, CLASS_NAMES
from model import build_model
from settings import PROFILES


parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=PROFILES, default="teaching")
parser.add_argument("--strategy", choices=["frozen", "partial"], required=True)
args = parser.parse_args()

checkpoint = Path("outputs") / f"{args.profile}_{args.strategy}" / "best_model.pth"
model = build_model(args.strategy)
state_dict = torch.load(
    checkpoint, map_location="cpu", weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

loaders, _ = build_loaders("data", PROFILES[args.profile])
images, labels = next(iter(loaders["test"]))
with torch.inference_mode():
    predictions = model(images[:4]).argmax(dim=1)

print("checkpoint_loaded=", checkpoint)
for true, pred in zip(labels[:4], predictions):
    print(f"true={CLASS_NAMES[true]} predicted={CLASS_NAMES[pred]}")
