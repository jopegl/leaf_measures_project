"""Log CSV das validações (mesmo formato do training_log.csv antigo) e prints."""
from __future__ import annotations

import csv
import os
from datetime import datetime

import torch

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


class TrainWindow:
    """
    Médias de treino desde a última linha registrada (janela de log_every iterações):
    loss total, loss por medida, MAE e RER por medida (nas unidades reais, desnormalizadas).
    Acumula em tensores no próprio device e só sincroniza com a CPU ao ler as médias.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.loss = self.parts = self.abs_err = self.rel_err = None

    def add(self, loss, per_comp, out, target):
        from leaf_measures.data.normalization import denormalize

        with torch.no_grad():
            pred = torch.stack([denormalize(m, out[:, i]) for i, m in enumerate(MEASURE_ORDER)], dim=1)
            real = torch.stack([denormalize(m, target[:, i]) for i, m in enumerate(MEASURE_ORDER)], dim=1)
            abs_err = (pred - real).abs().mean(dim=0)
            rel_err = ((pred - real).abs() / (real.abs() + 1e-8) * 100.0).mean(dim=0)
            vals = (loss.detach(), per_comp.detach(), abs_err, rel_err)
            if self.n == 0:
                self.loss, self.parts, self.abs_err, self.rel_err = (v.clone() for v in vals)
            else:
                self.loss += vals[0]
                self.parts += vals[1]
                self.abs_err += vals[2]
                self.rel_err += vals[3]
        self.n += 1

    def averages(self) -> dict:
        n = max(self.n, 1)
        parts, mae, rer = ((t / n).tolist() for t in (self.parts, self.abs_err, self.rel_err))
        return {
            "loss": (self.loss / n).item(),
            "loss_parts": dict(zip(MEASURE_ORDER, parts)),
            "mae": dict(zip(MEASURE_ORDER, mae)),
            "rer": dict(zip(MEASURE_ORDER, rer)),
        }
