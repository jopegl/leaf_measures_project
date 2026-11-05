"""
Testes da análise de erros: usa o dataset falso do conftest e predições
sintéticas com um padrão plantado (erro maior quando a folha é pequena na foto).
Não precisa de GPU nem de modelo treinado.
"""
import os

import numpy as np
import pandas as pd
import pytest

from conftest import SIZE

MEASURES = ["length", "width", "perimeter", "area"]


def _write_predictions(results_dir, exp, features, pattern=True, seed=0):
    rng = np.random.default_rng(seed)
    names = list(features.index)
    for cv in (1, 2):
        fold_dir = os.path.join(results_dir, exp, f"cv{cv}_split1", "predicoes")
        os.makedirs(fold_dir, exist_ok=True)
        for m in MEASURES:
            rows = []
            for n in names:
                real = features.loc[n, f"leaf_{m}"]
                frac = features.loc[n, "leaf_frac"]
                rel = 0.02 + (0.3 * (frac < 25) if pattern else 0)  # folha pequena -> erro grande
                rows.append((n, real * (1 + rel * rng.uniform(0.8, 1.2)), real))
            pd.DataFrame(rows, columns=["name", "predicted", "real"]).to_csv(
                os.path.join(fold_dir, f"{m}_predictions.csv"), index=False)


@pytest.fixture
def varied_dataset(fake_dataset):
    """Reescreve as máscaras com folhas de tamanhos diferentes."""
    root = fake_dataset["root"]
    rng = np.random.default_rng(3)
    for leaf in sorted(os.listdir(root)):
        seg_dir = os.path.join(root, leaf, "segmentation")
        for f in sorted(os.listdir(seg_dir)):
            r = int(rng.integers(6, 30))
            yy, xx = np.mgrid[:SIZE, :SIZE]
            seg = np.zeros((SIZE, SIZE), np.uint8)
            seg[(yy - 30) ** 2 + (xx - 30) ** 2 < r ** 2] = 1
            seg[50:58, 50:58] = 2
            seg.tofile(os.path.join(seg_dir, f))
    return fake_dataset


def test_features_cover_dataset(varied_dataset):
    from leaf_measures.analysis.features import compute_features

    feats = compute_features(varied_dataset["root"], SIZE, SIZE, verbose=False)
    assert len(feats) == len(varied_dataset["ids"])
    assert feats.index[0].endswith(".jpg") and "/" in feats.index[0]
    for col in ("leaf_frac", "sharpness", "brightness", "geom_area_rer", "leaf_touches_border"):
        assert feats[col].notna().all(), col


def test_per_image_aggregates_folds(varied_dataset, tmp_path):
    from leaf_measures.analysis.features import compute_features
    from leaf_measures.analysis.ranking import load_predictions, per_image_errors

    feats = compute_features(varied_dataset["root"], SIZE, SIZE, verbose=False)
    _write_predictions(str(tmp_path / "results"), "exp", feats)
    table = per_image_errors(load_predictions(str(tmp_path / "results"), "exp"))
    assert len(table) == len(feats)
    assert (table["n_folds"] == 2).all()
    assert {"area_rer", "area_error", "area_abs_error", "rer_mean"} <= set(table.columns)


def test_planted_pattern_is_found(varied_dataset, tmp_path):
    from leaf_measures.analysis.features import compute_features
    from leaf_measures.analysis.patterns import find_clues
    from leaf_measures.analysis.ranking import load_predictions, per_image_errors

    feats = compute_features(varied_dataset["root"], SIZE, SIZE, verbose=False)
    _write_predictions(str(tmp_path / "results"), "exp", feats)
    table = per_image_errors(load_predictions(str(tmp_path / "results"), "exp")).join(feats)
    clues, _ = find_clues(table, quantile=0.25)
    leaf_frac = [c for c in clues if c.feature == "leaf_frac"]
    assert leaf_frac, [c.text for c in clues]
    assert leaf_frac[0].stats["effect"] < 0          # erro maior quando a folha é MENOR
    assert leaf_frac[0].strength in ("forte", "moderado")


def test_no_pattern_no_strong_clues(varied_dataset, tmp_path):
    from leaf_measures.analysis.features import compute_features
    from leaf_measures.analysis.patterns import find_clues
    from leaf_measures.analysis.ranking import load_predictions, per_image_errors

    feats = compute_features(varied_dataset["root"], SIZE, SIZE, verbose=False)
    _write_predictions(str(tmp_path / "results"), "exp", feats, pattern=False)
    table = per_image_errors(load_predictions(str(tmp_path / "results"), "exp")).join(feats)
    clues, _ = find_clues(table, quantile=0.25)
    assert not [c for c in clues if c.kind == "caracteristica" and c.strength == "forte"]


def test_end_to_end_two_models(varied_dataset, tmp_path):
    from leaf_measures.analysis import analyze
    from leaf_measures.analysis.features import compute_features
    from leaf_measures.config import Config

    feats = compute_features(varied_dataset["root"], SIZE, SIZE, verbose=False)
    res = str(tmp_path / "results")
    _write_predictions(res, "a", feats, seed=1)
    _write_predictions(res, "b", feats, seed=2)
    data_cfg = Config({"root": varied_dataset["root"], "height": SIZE, "width": SIZE})
    analyze(["a", "b"], data_cfg, results_dir=res, top_n=4, quantile=0.25)

    out = os.path.join(res, "analise")
    for f in ("features.csv", "a/indicios.md", "a/imagens_erros.csv", "a/piores_area.csv",
              "a/imagens/piores_mean.png", "a/graficos/vies_erro_vs_real.png",
              "comparacao_a_vs_b/indicios_comparacao.md", "comparacao_a_vs_b/erro_por_imagem_entre_modelos.csv"):
        assert os.path.isfile(os.path.join(out, f)), f
    assert "fração da imagem ocupada pela folha" in open(os.path.join(out, "a", "indicios.md"), encoding="utf-8").read()
