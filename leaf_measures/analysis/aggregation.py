"""
Novas formas de predição a partir das predições já salvas pelo evaluate.py,
com as métricas (MAE, RMSE, RER, R²) RECALCULADAS sobre a nova predição:

  - por FOLHA: a predição de uma folha é a média (ou mediana) das predições de
    todas as imagens dela no conjunto de teste do fold;
  - ENSEMBLE: a predição de uma imagem é a média (ou mediana) das predições de
    várias redes para essa imagem;
  - os dois combinados: primeiro o ensemble por imagem, depois a agregação por folha.

O resultado é gravado como um "experimento virtual" em results/<nome>/, no mesmo
formato do evaluate.py (predicoes/, deepviewer_test.csv, summary.json,
resumo_rer.csv, resumo_metricas.csv). Assim compare.py, analyze.py e o Deep
Viewer funcionam com ele sem nada especial.

As médias por CV e total continuam ponderadas pelo nº de amostras do fold
(imagens no caso do ensemble; FOLHAS no caso da agregação por folha).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from leaf_measures.analysis.ranking import load_predictions
from leaf_measures.data.normalization import MEASURE_ORDER
from leaf_measures.deep_viewer import TEST_SHEET, write_test_sheet

AGG_FUNCS = {"mean": "média", "median": "mediana"}
LEAF_KEYS = ("pasta", "pasta+medidas")
EPS = 1e-8


# ------------------------------------------------------------------ leitura
def fold_tables(results_dir: str, experiment: str, folds=None) -> dict[str, pd.DataFrame]:
    """{fold: DataFrame indexado por 'name' com true_<m> e pred_<m>}."""
    long = load_predictions(results_dir, experiment, folds)
    out = {}
    for fold, df in long.groupby("fold"):
        wide = df.pivot_table(index="name", columns="measure", values=["real", "predicted"], aggfunc="first")
        t = pd.DataFrame(index=wide.index)
        for m in MEASURE_ORDER:
            t[f"true_{m}"] = wide[("real", m)]
            t[f"pred_{m}"] = wide[("predicted", m)]
        out[fold] = t.dropna()
    return out


# ------------------------------------------------------------------ métricas
def metrics_np(pred: np.ndarray, true: np.ndarray) -> dict:
    """Mesmas fórmulas de evaluation/metrics.py::calculate_measure_metrics, em numpy."""
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    err = pred - true
    ss_res = (err ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "rer": float((np.abs(err) / (np.abs(true) + EPS) * 100.0).mean()),
        "r2": float(1.0 - ss_res / (ss_tot + EPS)),
    }


def table_metrics(t: pd.DataFrame) -> dict:
    return {m: metrics_np(t[f"pred_{m}"], t[f"true_{m}"]) for m in MEASURE_ORDER}


# ------------------------------------------------------------------ agregações
def ensemble(tables_by_exp: dict[str, dict[str, pd.DataFrame]], how: str = "mean") -> tuple[dict, list[str]]:
    """
    tables_by_exp: {experimento: {fold: tabela}}. Usa só os folds avaliados em
    TODOS os experimentos e, em cada fold, só as imagens presentes em todos.
    """
    exps = list(tables_by_exp)
    common_folds = sorted(set.intersection(*(set(t) for t in tables_by_exp.values())))
    warnings = []
    skipped = sorted(set.union(*(set(t) for t in tables_by_exp.values())) - set(common_folds))
    if skipped:
        warnings.append(f"Folds ignorados por não estarem avaliados em todos os experimentos: {skipped}")

    out = {}
    for fold in common_folds:
        tabs = [tables_by_exp[e][fold] for e in exps]
        names = sorted(set.intersection(*(set(t.index) for t in tabs)))
        lost = max(len(t) for t in tabs) - len(names)
        if lost:
            warnings.append(f"{fold}: {lost} imagem(ns) não estão em todos os experimentos e foram ignoradas")
        res = pd.DataFrame(index=pd.Index(names, name="name"))
        for m in MEASURE_ORDER:
            trues = np.stack([t.loc[names, f"true_{m}"].to_numpy() for t in tabs])
            if np.abs(trues - trues[0]).max() > 1e-4:
                raise ValueError(f"{fold}/{m}: os valores reais diferem entre os experimentos "
                                 "(folds ou dataset diferentes?). Ensemble cancelado.")
            preds = np.stack([t.loc[names, f"pred_{m}"].to_numpy() for t in tabs])
            res[f"true_{m}"] = trues[0]
            res[f"pred_{m}"] = preds.mean(axis=0) if how == "mean" else np.median(preds, axis=0)
        out[fold] = res
    return out, warnings


def leaf_id(name: str) -> str:
    """'leaf_07/imagem.jpg' -> 'leaf_07' (a pasta é a folha)."""
    return name.replace("\\", "/").split("/")[0]


def by_leaf(t: pd.DataFrame, how: str = "mean", key: str = "pasta") -> tuple[pd.DataFrame, float]:
    """
    Agrega as imagens de cada folha. Retorna (tabela por folha, maior variação
    relativa (%) do valor real entre imagens de uma mesma folha).

    key='pasta'          -> folha = pasta da imagem
    key='pasta+medidas'  -> folha = pasta + valores reais (separa, numa mesma pasta,
                            imagens com medidas anotadas diferentes)
    """
    if key not in LEAF_KEYS:
        raise ValueError(f"leaf_key deve ser um de {LEAF_KEYS}")
    folder = pd.Series(t.index.map(leaf_id), index=t.index)
    if key == "pasta":
        group = folder
    else:
        sig = t[[f"true_{m}" for m in MEASURE_ORDER]].round(4).astype(str).agg("|".join, axis=1)
        codes = sig.groupby(folder).transform(lambda s: pd.factorize(s)[0] + 1)
        n_sets = codes.groupby(folder).transform("max")
        # pasta com um único conjunto de medidas mantém o nome simples; senão, pasta#1, pasta#2...
        group = folder.where(n_sets == 1, folder + "#" + codes.astype(str))

    agg = "mean" if how == "mean" else "median"
    g = t.groupby(group.values)
    res = pd.DataFrame(index=pd.Index(sorted(g.groups), name="name"))
    worst_var = 0.0
    for m in MEASURE_ORDER:
        tr = g[f"true_{m}"]
        rel = (tr.max() - tr.min()) / tr.mean().abs().clip(lower=EPS) * 100
        worst_var = max(worst_var, float(rel.max()))
        res[f"true_{m}"] = tr.mean()
        res[f"pred_{m}"] = g[f"pred_{m}"].agg(agg)
    res["n_imagens"] = g.size()
    return res, worst_var


# ------------------------------------------------------------------ saída
def write_virtual_experiment(name: str, label: str, fold_tabs: dict, results_dir: str, meta: dict) -> dict:
    from leaf_measures.evaluation.evaluator import write_long_metrics, write_rer_table
    from leaf_measures.evaluation.metrics import weighted_average

    out_root = os.path.join(results_dir, name)
    os.makedirs(out_root, exist_ok=True)
    folds = []
    for fold, t in sorted(fold_tabs.items()):
        cv, split = (int(x) for x in fold.replace("cv", "").split("_split"))
        fdir = os.path.join(out_root, fold)
        os.makedirs(os.path.join(fdir, "predicoes"), exist_ok=True)
        for m in MEASURE_ORDER:
            pd.DataFrame({"name": t.index, "predicted": t[f"pred_{m}"].to_numpy(),
                          "real": t[f"true_{m}"].to_numpy()}).to_csv(
                os.path.join(fdir, "predicoes", f"{m}_predictions.csv"), index=False)
        write_test_sheet(os.path.join(fdir, TEST_SHEET), list(t.index),
                         {m: t[f"pred_{m}"].to_numpy() for m in MEASURE_ORDER},
                         {m: t[f"true_{m}"].to_numpy() for m in MEASURE_ORDER})
        if "n_imagens" in t:
            t[["n_imagens"]].to_csv(os.path.join(fdir, "imagens_por_folha.csv"))
        folds.append({"fold": fold, "cv": cv, "split": split, "n_samples": len(t), "metrics": table_metrics(t)})

    by_cv: dict = {}
    for r in folds:
        by_cv.setdefault(r["cv"], []).append(r)
    cv_avg = {str(cv): {"n_samples": sum(e["n_samples"] for e in es), "n_folds": len(es),
                        "metrics": weighted_average(es)} for cv, es in sorted(by_cv.items())}
    overall = {"n_samples": sum(r["n_samples"] for r in folds), "metrics": weighted_average(folds)}
    summary = {"experiment": name, "label": label, "backbone": meta.get("backbone", "—"),
               "input_mode": meta.get("input_mode", "—"), "aggregation": meta,
               "folds": folds, "cv_averages": cv_avg, "overall": overall}
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_rer_table(summary, os.path.join(out_root, "resumo_rer.csv"))
    write_long_metrics(summary, os.path.join(out_root, "resumo_metricas.csv"))
    return summary


def comparison_table(summary: dict, base_summaries: dict) -> pd.DataFrame:
    """RER total (média ponderada) por medida: cada experimento-base x a nova predição."""
    rows = []
    for lab, s in list(base_summaries.items()) + [(summary["label"], summary)]:
        row = {"predicao": lab, "n_amostras": s["overall"]["n_samples"]}
        for m in MEASURE_ORDER:
            row[f"rer_{m}"] = s["overall"]["metrics"][m]["rer"]
        row["rer_medio"] = float(np.mean([row[f"rer_{m}"] for m in MEASURE_ORDER]))
        rows.append(row)
    return pd.DataFrame(rows)


def base_summary_from_tables(name: str, tabs: dict) -> dict:
    """Recalcula o resumo de um experimento-base só nos folds/imagens usados (comparação justa)."""
    from leaf_measures.evaluation.metrics import weighted_average

    folds = [{"n_samples": len(t), "metrics": table_metrics(t)} for t in tabs.values()]
    return {"label": name, "overall": {"n_samples": sum(f["n_samples"] for f in folds),
                                       "metrics": weighted_average(folds)}}


def aggregate(experiments: list[str], results_dir: str = "results", ensemble_how: str | None = None,
              leaf_how: str | None = None, leaf_key: str = "pasta", folds=None, name: str | None = None) -> dict:
    if not ensemble_how and not leaf_how:
        raise ValueError("Escolha pelo menos uma agregação: --ensemble e/ou --leaf")
    if ensemble_how and len(experiments) < 2:
        raise ValueError("O ensemble precisa de pelo menos 2 experimentos")
    if not ensemble_how and len(experiments) != 1:
        raise ValueError("Para agregar por folha vários experimentos juntos, use também --ensemble")

    tables = {e: fold_tables(results_dir, e, folds) for e in experiments}
    warnings = []
    if ensemble_how:
        current, w = ensemble(tables, ensemble_how)
        warnings += w
        auto = f"ensemble_{AGG_FUNCS[ensemble_how].replace('é', 'e')}__" + "+".join(experiments)
        label = f"Ensemble ({AGG_FUNCS[ensemble_how]}) de {len(experiments)} redes"
    else:
        current = tables[experiments[0]]
        auto, label = experiments[0], experiments[0]
    # bases para comparação: cada experimento restrito aos mesmos folds/imagens
    bases = {e: base_summary_from_tables(e, {f: tables[e][f].loc[current[f].index] for f in current})
             for e in experiments}
    if ensemble_how and leaf_how:  # o ensemble por imagem também serve de referência para a versão por folha
        bases[f"{label} — por imagem"] = base_summary_from_tables("ens", current)

    if leaf_how:
        leaf_tabs, worst = {}, 0.0
        for f, t in current.items():
            leaf_tabs[f], v = by_leaf(t, leaf_how, leaf_key)
            worst = max(worst, v)
        if worst > 1.0 and leaf_key == "pasta":
            warnings.append(
                f"As medidas REAIS variam até {worst:.1f}% entre imagens da mesma pasta. A folha recebe a média "
                "dos valores reais. Se a pasta reúne a mesma folha em momentos diferentes (com tamanhos "
                "diferentes), rode com --leaf-key pasta+medidas.")
        current = leaf_tabs
        auto += f"__folha_{AGG_FUNCS[leaf_how].replace('é', 'e')}"
        label += f" — por folha ({AGG_FUNCS[leaf_how]})"

    name = name or auto
    meta = {"experimentos": experiments, "ensemble": ensemble_how, "folha": leaf_how,
            "folha_chave": leaf_key if leaf_how else None}
    summary = write_virtual_experiment(name, label, current, results_dir, meta)
    comp = comparison_table(summary, bases)
    comp.to_csv(os.path.join(results_dir, name, "comparacao_rer.csv"), index=False)
    summary["comparison"] = comp
    summary["warnings"] = warnings
    return summary
