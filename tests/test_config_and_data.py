"""Testes que não dependem de GPU nem de pesos pré-treinados."""
import os

import pytest

from leaf_measures.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "configs", "experiments")


@pytest.mark.parametrize("name", sorted(f for f in os.listdir(EXP) if f.endswith(".yaml")))
def test_all_experiment_configs_load(name):
    cfg = load_config(os.path.join(EXP, name))
    assert cfg.experiment.name == name[:-5]
    assert isinstance(cfg.train.lr_head, float)
    assert len(cfg.loss.weights) == 4


def test_inheritance_chain_and_overrides():
    cfg = load_config(os.path.join(EXP, "convnext_tiny_hadamard.yaml"),
                      ["train.lr_head=3e-4", "folds.cvs=[2]", "model.head.dropout=0.1"])
    assert cfg.input.mode == "hadamard"          # herdado de resnet50_hadamard.yaml
    assert cfg.model.backbone == "convnext_tiny"
    assert cfg.train.lr_head == pytest.approx(3e-4)
    assert cfg.folds.cvs == [2]
    assert cfg.model.head.hidden_dims == [256]   # herdado da base
    assert cfg.model.head.dropout == 0.1


def test_folds_and_splits(fake_dataset):
    from leaf_measures.data.folds import discover_folds, folds_from_config, parse_fold_id
    from leaf_measures.data.splits import load_split_ids

    assert parse_fold_id("cv3_split4") == (3, 4)
    with pytest.raises(ValueError):
        parse_fold_id("cv3-4")
    folds = discover_folds(fake_dataset["cv_root"])
    assert [f.id for f in folds] == ["cv1_split1"]
    ids = load_split_ids(folds[0].csv("test"))
    assert ids == set(fake_dataset["ids"][10:12])

    cfg = load_config(os.path.join(EXP, "resnet50_rgb.yaml"), [f"data.cv_root={fake_dataset['cv_root']}"])
    assert len(folds_from_config(cfg)) == 25
    assert [f.id for f in folds_from_config(cfg, ["cv2_split5"])] == ["cv2_split5"]


def test_metrics_weighted_average():
    from leaf_measures.evaluation.metrics import weighted_average

    a = {"n_samples": 1, "metrics": {"area": {"mae": 1.0, "rmse": 1.0, "rer": 10.0, "r2": 0.5}}}
    b = {"n_samples": 3, "metrics": {"area": {"mae": 3.0, "rmse": 3.0, "rer": 30.0, "r2": 0.9}}}
    avg = weighted_average([a, b])["area"]
    assert avg["mae"] == pytest.approx(2.5)
    assert avg["rer"] == pytest.approx(25.0)
