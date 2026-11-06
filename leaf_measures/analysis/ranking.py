"""
Erro por imagem a partir dos CSVs salvos pelo evaluate.py
(results/<exp>/<fold>/predicoes/<medida>_predictions.csv).

Numa cross-validation repetida, a mesma imagem cai no teste de vários folds
(em geral uma vez por CV). Aqui juntamos essas ocorrências e usamos a média
por imagem, o que torna o ranking mais estável do que olhar um fold só.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from leaf_measures.data.normalization import MEASURE_ORDER

_FOLD_RE = re.compile(r"^cv(\d+)_split(\d+)$")


def load_predictions(results_dir: str, experiment: str, folds: list[str] | None = None) -> pd.DataFrame:
    """Formato longo: uma linha por (fold, imagem, medida)."""
    exp_dir = os.path.join(results_dir, experiment)
    frames = []
    for fold_dir in sorted(glob.glob(os.path.join(exp_dir, "cv*_split*"))):
        fold = os.path.basename(fold_dir)
        if not _FOLD_RE.match(fold) or (folds and fold not in folds):
            continue
        for m in MEASURE_ORDER:
            path = os.path.join(fold_dir, "predicoes", f"{m}_predictions.csv")
            if os.path.isfile(path):
                df = pd.read_csv(path)
                df["fold"], df["measure"] = fold, m
                frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"Nenhuma predição encontrada em {exp_dir}/<fold>/predicoes/. "
            f"Rode antes: python evaluate.py -c configs/experiments/{experiment}.yaml"
        )
    df = pd.concat(frames, ignore_index=True)
    df["error"] = df["predicted"] - df["real"]
    df["abs_error"] = df["error"].abs()
    df["rer"] = df["abs_error"] / df["real"].abs().clip(lower=1e-8) * 100.0
    return df


def per_image_errors(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Uma linha por imagem; colunas por medida:
      <m>_real, <m>_pred, <m>_error (com sinal), <m>_abs_error, <m>_rer, <m>_rer_std
    + rer_mean (média do RER nas 4 medidas), n_folds.
    """
    g = preds.groupby(["name", "measure"])
    agg = g.agg(real=("real", "first"), pred=("predicted", "mean"), error=("error", "mean"),
                abs_error=("abs_error", "mean"), rer=("rer", "mean"), rer_std=("rer", "std"),
                n_folds=("fold", "nunique"))
    wide = agg.unstack("measure")
    wide.columns = [f"{m}_{col}" for col, m in wide.columns]
    out = pd.DataFrame(index=wide.index)
    for m in MEASURE_ORDER:
        for col in ("real", "pred", "error", "abs_error", "rer", "rer_std"):
            key = f"{m}_{col}"
            if key in wide:
                out[key] = wide[key]
    rer_cols = [f"{m}_rer" for m in MEASURE_ORDER if f"{m}_rer" in out]
    out["rer_mean"] = out[rer_cols].mean(axis=1)
    n_cols = [c for c in wide.columns if c.endswith("_n_folds")]
    out["n_folds"] = wide[n_cols].max(axis=1).astype(int)
    return out


def rank_images(table: pd.DataFrame, target: str, n: int, worst: bool = True) -> pd.DataFrame:
    """`target` = '<medida>_rer' ou 'rer_mean'."""
    return table.sort_values(target, ascending=not worst).head(n)


def worst_mask(table: pd.DataFrame, target: str, quantile: float) -> pd.Series:
    """True para as imagens no pior `quantile` (ex.: 0.1 = 10% piores) de `target`."""
    thr = table[target].quantile(1.0 - quantile)
    return table[target] >= thr


def multi_measure_outliers(table: pd.DataFrame, quantile: float, min_measures: int = 3) -> pd.DataFrame:
    """Imagens que estão entre as piores em >= min_measures medidas ao mesmo tempo."""
    flags = pd.DataFrame({m: worst_mask(table, f"{m}_rer", quantile) for m in MEASURE_ORDER
                          if f"{m}_rer" in table})
    count = flags.sum(axis=1)
    out = table.loc[count >= min_measures].copy()
    out["n_medidas_piores"] = count[count >= min_measures]
    pt = {"length": "comprimento", "width": "largura", "perimeter": "perímetro", "area": "área"}
    out["medidas_piores"] = flags.loc[out.index].apply(lambda r: ", ".join(pt[c] for c in flags if r[c]), axis=1)
    return out.sort_values(["n_medidas_piores", "rer_mean"], ascending=False)


def cross_model(tables: dict, quantile: float) -> pd.DataFrame:
    """
    tables: {experimento: per_image_errors(...)}.
    Retorna por imagem o rer_mean de cada modelo, em quantos modelos ela está
    entre as piores, e uma classificação:
      'difícil para todos'  -> provável problema da imagem/anotação
      'difícil só para X'   -> provável limitação daquele modelo
    """
    common = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    df = pd.DataFrame({exp: t.loc[common, "rer_mean"] for exp, t in tables.items()})
    worst = pd.DataFrame({exp: worst_mask(t.loc[common], "rer_mean", quantile) for exp, t in tables.items()})
    df["n_modelos_piores"] = worst.sum(axis=1)
    n = len(tables)

    def classify(row):
        k = row["n_modelos_piores"]
        if k == n:
            return "difícil para todos"
        if k == 1:
            return "difícil só para " + next(e for e in tables if worst.loc[row.name, e])
        if k > 1:
            return f"difícil para {k} de {n}"
        return ""

    df["classificacao"] = df.apply(classify, axis=1)
    df["rer_medio_modelos"] = df[list(tables)].mean(axis=1)
    return df.sort_values(["n_modelos_piores", "rer_medio_modelos"], ascending=False)


def agreement(tables: dict) -> pd.DataFrame:
    """Correlação de Spearman do rer_mean por imagem entre pares de modelos."""
    common = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    df = pd.DataFrame({e: t.loc[common, "rer_mean"] for e, t in tables.items()})
    return df.rank().corr()

