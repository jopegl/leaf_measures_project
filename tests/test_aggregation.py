"""Predição por folha e ensemble: os números são conferidos com cálculo direto em pandas."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from leaf_measures.analysis.aggregation import aggregate, by_leaf

M = ["length", "width", "perimeter", "area"]
FOLDS = ["cv1_split1", "cv1_split2", "cv2_split1"]


@pytest.fixture
def results(tmp_path):
    rng = np.random.default_rng(0)
    leaves = {f"leaf_{i:02d}": {m: rng.uniform(5, 30) for m in M} for i in range(8)}
    res = tmp_path / "results"
    for exp, noise in (("A", 0.10), ("B", 0.05)):
        for k, fold in enumerate(FOLDS):
            d = res / exp / fold / "predicoes"
            d.mkdir(parents=True)
            sel = list(leaves)[k::2] if k < 2 else list(leaves)
            for m in M:
                rows = [(f"{l}/{l}_img{j}.jpg", leaves[l][m] * (1 + rng.normal(0, noise)), leaves[l][m])
                        for l in sel for j in range(3)]
                pd.DataFrame(rows, columns=["name", "predicted", "real"]).to_csv(d / f"{m}_predictions.csv", index=False)
    return str(res)


def _load(res, exp, fold, m):
    return pd.read_csv(os.path.join(res, exp, fold, "predicoes", f"{m}_predictions.csv")).set_index("name")


def _summary(res, name):
    with open(os.path.join(res, name, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("how,fn", [("mean", "mean"), ("median", "median")])
def test_leaf_rer_matches_manual(results, how, fn):
    s = aggregate(["A"], results, leaf_how=how)
    total, n = 0.0, 0
    for fold in FOLDS:
        d = _load(results, "A", fold, "area")
        g = d.groupby(d.index.str.split("/").str[0]).agg(p=("predicted", fn), r=("real", "mean"))
        total += ((g.p - g.r).abs() / g.r * 100).mean() * len(g)
        n += len(g)
    assert s["overall"]["n_samples"] == n
    assert s["overall"]["metrics"]["area"]["rer"] == pytest.approx(total / n)


def test_ensemble_rer_matches_manual(results):
    s = aggregate(["A", "B"], results, ensemble_how="mean")
    total, n = 0.0, 0
    for fold in FOLDS:
        a, b = _load(results, "A", fold, "width"), _load(results, "B", fold, "width")
        p = (a.predicted + b.predicted.loc[a.index]) / 2
        total += ((p - a.real).abs() / a.real * 100).mean() * len(a)
        n += len(a)
    assert s["overall"]["metrics"]["width"]["rer"] == pytest.approx(total / n)
    # o ensemble de duas redes deve ficar entre/abaixo das duas, nunca muito pior que a melhor
    comp = s["comparison"].set_index("predicao")
    assert comp.loc[s["label"], "rer_medio"] <= comp.loc["A", "rer_medio"]


def test_virtual_experiment_has_evaluate_layout(results):
    s = aggregate(["A", "B"], results, ensemble_how="mean", leaf_how="mean", name="ens_folha")
    base = os.path.join(results, "ens_folha")
    for f in ("summary.json", "resumo_rer.csv", "resumo_metricas.csv", "comparacao_rer.csv",
              "cv1_split1/predicoes/area_predictions.csv", "cv1_split1/deepviewer_test.csv",
              "cv1_split1/imagens_por_folha.csv"):
        assert os.path.isfile(os.path.join(base, f)), f
    assert _summary(results, "ens_folha")["aggregation"]["ensemble"] == "mean"
    assert len(s["comparison"]) == 4   # A, B, ensemble por imagem, ensemble por folha


def test_leaf_key_pasta_mais_medidas_splits_different_reals():
    t = pd.DataFrame(
        {**{f"true_{m}": [10, 10, 20, 20] for m in M}, **{f"pred_{m}": [11, 9, 21, 19] for m in M}},
        index=["leaf_01/a.jpg", "leaf_01/b.jpg", "leaf_01/c.jpg", "leaf_01/d.jpg"])
    by_folder, var = by_leaf(t, "mean", "pasta")
    assert len(by_folder) == 1 and var > 50     # reais diferentes na mesma pasta: avisa
    split, _ = by_leaf(t, "mean", "pasta+medidas")
    assert sorted(split.index) == ["leaf_01#1", "leaf_01#2"]
    assert list(split["pred_area"]) == [10, 20]


def test_invalid_combinations(results):
    with pytest.raises(ValueError):
        aggregate(["A"], results)                              # nenhuma agregação
    with pytest.raises(ValueError):
        aggregate(["A"], results, ensemble_how="mean")          # ensemble de 1 rede
    with pytest.raises(ValueError):
        aggregate(["A", "B"], results, leaf_how="mean")         # várias redes sem --ensemble


def test_ensemble_refuses_mismatched_ground_truth(results, tmp_path):
    p = os.path.join(results, "B", "cv1_split1", "predicoes", "area_predictions.csv")
    d = pd.read_csv(p)
    d["real"] += 1.0
    d.to_csv(p, index=False)
    with pytest.raises(ValueError, match="reais diferem"):
        aggregate(["A", "B"], results, ensemble_how="mean")
