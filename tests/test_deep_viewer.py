"""Formato das planilhas do Deep Viewer (não precisa de GPU)."""
import csv

import pytest

from leaf_measures.deep_viewer import TrainSheetWriter, train_columns, write_test_sheet
from leaf_measures.deep_viewer import test_columns as dv_test_columns

M = ["length", "width", "perimeter", "area"]
PARTS = {m: 0.1 for m in M}
METRICS = {m: {"mae": 0.2, "rmse": 0.3, "rer": 4.0, "r2": 0.9} for m in M}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _log(w, it, val=False):
    w.log_train(it, 1.0 / it, 1e-3, PARTS, {m: 0.5 for m in M}, {m: 5.0 for m in M})
    if val:
        w.log_val(it, 0.9 / it, PARTS, METRICS)


def test_columns_follow_deep_viewer_convention():
    cols = train_columns()
    assert cols[:4] == ["iteration", "train_loss", "val_loss", "learning_rate"]
    assert all(c.startswith(("train_", "val_")) for c in cols[4:])
    # cada métrica de treino tem a de validação com o MESMO nome depois do prefixo
    for c in cols:
        if c.startswith("train_") and c != "train_loss":
            assert "val_" + c[len("train_"):] in cols, c
    assert dv_test_columns() == ["sample_id"] + [f"{p}_{m}" for m in M for p in ("true", "pred")]


def test_train_sheet_rows_and_val_cells(tmp_path):
    path = tmp_path / "dv.csv"
    w = TrainSheetWriter(str(path))
    _log(w, 100)
    _log(w, 200, val=True)
    w.flush()
    rows = _rows(path)
    assert [r["iteration"] for r in rows] == ["100", "200"]
    assert rows[0]["val_loss"] == "" and rows[0]["train_loss"] != ""   # sem validação: val vazio
    assert float(rows[1]["val_rer_area"]) == 4.0
    assert "," not in rows[1]["train_loss"]                               # decimal com ponto


def test_resume_drops_rows_after_checkpoint(tmp_path):
    path = tmp_path / "dv.csv"
    w = TrainSheetWriter(str(path))
    for it in (100, 200, 300):
        _log(w, it, val=(it == 200))
    w.flush()                       # simula: checkpoint na 200, mas o CSV chegou a 300 antes de cair
    w2 = TrainSheetWriter(str(path), start_iteration=200)
    _log(w2, 300)
    _log(w2, 400, val=True)
    w2.flush()
    its = [int(r["iteration"]) for r in _rows(path)]
    assert its == [100, 200, 300, 400]   # sem iteração repetida


def test_fresh_start_overwrites(tmp_path):
    path = tmp_path / "dv.csv"
    w = TrainSheetWriter(str(path))
    _log(w, 100)
    w.flush()
    TrainSheetWriter(str(path), start_iteration=0).flush()
    assert _rows(path) == []


def test_row_without_train_loss_is_refused(tmp_path):
    w = TrainSheetWriter(str(tmp_path / "dv.csv"))
    w.log_val(500, 0.5, PARTS, METRICS)
    with pytest.raises(RuntimeError):
        w.flush()


def test_test_sheet(tmp_path):
    path = tmp_path / "t.csv"
    write_test_sheet(str(path), ["a/1.jpg", "a/2.jpg"], {m: [1.0, 2.0] for m in M}, {m: [1.5, 2.5] for m in M})
    rows = _rows(path)
    assert rows[1]["sample_id"] == "a/2.jpg" and rows[1]["true_area"] == "2.5" and rows[1]["pred_area"] == "2.0"
