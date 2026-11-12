"""
Exportação no formato de planilhas do Deep Viewer.

Cada fold vira um "experimento" do Deep Viewer, com unidade de passo = iteração:

  runs/<exp>/<fold>/deepviewer_train.csv   planilha de TREINO
  results/<exp>/<fold>/deepviewer_test.csv planilha de TESTE (multi-saída)

Planilha de treino (uma linha por iteração registrada):
  iteration, train_loss, val_loss, learning_rate,
  train_loss_<medida>, val_loss_<medida>      (loss de cada saída)
  train_mae_<medida>,  val_mae_<medida>
  train_rer_<medida>,  val_rer_<medida>
  val_rmse_<medida>,   val_r2_<medida>
- As colunas de treino são médias da janela desde a última linha (log_every iterações).
- As colunas val_* só são preenchidas nas iterações em que houve validação.
- Mesmo nome depois de train_/val_ (ex.: train_mae_area e val_mae_area) = mesmo gráfico.

Planilha de teste (uma linha por amostra; uma saída por medida):
  sample_id, true_length, pred_length, true_width, pred_width,
  true_perimeter, pred_perimeter, true_area, pred_area
"""
from __future__ import annotations

import csv
import os

from leaf_measures.data.normalization import MEASURE_ORDER

TRAIN_SHEET = "deepviewer_train.csv"
TEST_SHEET = "deepviewer_test.csv"


def train_columns() -> list[str]:
    cols = ["iteration", "train_loss", "val_loss", "learning_rate"]
    for prefix in ("train_loss", "val_loss", "train_mae", "val_mae", "train_rer", "val_rer", "val_rmse", "val_r2"):
        cols += [f"{prefix}_{m}" for m in MEASURE_ORDER]
    return cols


def test_columns() -> list[str]:
    cols = ["sample_id"]
    for m in MEASURE_ORDER:
        cols += [f"true_{m}", f"pred_{m}"]
    return cols


class TrainSheetWriter:
    """
    Acumula as linhas por iteração e grava em disco no `flush()`.

    O treino chama flush() junto com cada checkpoint, então o CSV nunca fica à
    frente do last.pth. Ao retomar de uma iteração N, as linhas > N (de uma
    execução que caiu depois do último checkpoint) são descartadas, garantindo
    que cada iteração apareça uma única vez, como o Deep Viewer exige.
    """

    def __init__(self, path: str, start_iteration: int = 0):
        self.path = path
        self.columns = train_columns()
        self.pending: dict[int, dict] = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        kept = []
        if start_iteration > 0 and os.path.isfile(path):
            with open(path, newline="", encoding="utf-8") as f:
                kept = [r for r in csv.DictReader(f) if r.get("iteration") and int(r["iteration"]) <= start_iteration]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.columns)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in self.columns} for r in kept)

    def _row(self, iteration: int) -> dict:
        return self.pending.setdefault(iteration, {"iteration": iteration})

    def log_train(self, iteration: int, loss: float, lr: float, loss_parts: dict, mae: dict, rer: dict):
        row = self._row(iteration)
        row.update(train_loss=loss, learning_rate=lr)
        for m in MEASURE_ORDER:
            row[f"train_loss_{m}"] = loss_parts[m]
            row[f"train_mae_{m}"] = mae[m]
            row[f"train_rer_{m}"] = rer[m]

    def log_val(self, iteration: int, loss: float, loss_parts: dict, metrics: dict):
        row = self._row(iteration)
        row["val_loss"] = loss
        for m in MEASURE_ORDER:
            row[f"val_loss_{m}"] = loss_parts[m]
            for k in ("mae", "rer", "rmse", "r2"):
                row[f"val_{k}_{m}"] = metrics[m][k]

    def flush(self):
        if not self.pending:
            return
        rows = [self.pending[i] for i in sorted(self.pending)]
        missing = [r["iteration"] for r in rows if "train_loss" not in r]
        if missing:  # o Deep Viewer exige train_loss em toda linha
            raise RuntimeError(f"Linhas sem train_loss nas iterações {missing[:5]}")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerows({k: r.get(k, "") for k in self.columns} for r in rows)
        self.pending.clear()


def write_test_sheet(path: str, names, preds: dict, targets: dict):
    """preds/targets: {medida: sequência de floats}, alinhados com `names`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(test_columns())
        for i, name in enumerate(names):
            row = [name]
            for m in MEASURE_ORDER:
                row += [float(targets[m][i]), float(preds[m][i])]
            w.writerow(row)
