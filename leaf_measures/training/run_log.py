"""Log CSV das validações (mesmo formato do training_log.csv antigo) e prints."""
from __future__ import annotations

import csv
import os
from datetime import datetime

from leaf_measures.data.normalization import MEASURE_ORDER


def init_log_csv(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        return
    header = ["split", "iteration", "timestamp", "lr", "val_loss"]
    for name in MEASURE_ORDER:
        header += [f"{name}_mae", f"{name}_rmse", f"{name}_rer", f"{name}_r2"]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def append_log_row(path: str, fold_id: str, iteration: int, lr: float, val_loss: float, metrics: dict):
    row = [fold_id, iteration, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), lr, val_loss]
    for name in MEASURE_ORDER:
        m = metrics[name]
        row += [m["mae"], m["rmse"], m["rer"], m["r2"]]
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def format_metrics(metrics: dict, indent: str = "  ") -> str:
    lines = []
    for name in MEASURE_ORDER:
        m = metrics[name]
        lines.append(
            f"{indent}{name.capitalize():<10} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
            f"RER={m['rer']:.2f}%  R²={m['r2']:.4f}"
        )
    return "\n".join(lines)


def print_val_block(title: str, val_loss: float, metrics: dict, iteration: int, iters: int, lr: float):
    print("\n" + "=" * 60)
    print(f"[{title}] Iteration [{iteration}/{iters}]  LR: {lr:.8f}  Loss: {val_loss:.4f}")
    print(format_metrics(metrics, indent="    "))
    print("=" * 60)


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")
