import csv
import json
from pathlib import Path


PROFILE = "teaching"
rows = []
for strategy in ["frozen", "partial"]:
    path = Path("outputs") / f"{PROFILE}_{strategy}" / "metrics.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "strategy": strategy,
        "trainable_parameters": result["trainable_parameters"],
        "train_seconds": round(result["train_seconds"], 1),
        "test_accuracy": round(result["test"]["accuracy"], 4),
        "macro_f1": round(result["test"]["macro_f1"], 4),
    })

with Path("outputs/strategy_comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("strategy | trainable | seconds | accuracy | macro_f1")
for row in rows:
    print(f"{row['strategy']:8s} | {row['trainable_parameters']:9d} | "
          f"{row['train_seconds']:7.1f} | {row['test_accuracy']:.4f} | {row['macro_f1']:.4f}")

