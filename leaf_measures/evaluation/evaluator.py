"""
Avaliação no TESTE de um experimento (todos os folds com best.pth).

Saída em results/<experimento>/:
  <fold>/graficos/*.png               (modo completo)
  <fold>/predicoes/<medida>_predictions.csv
  cv<N>/media_ponderada/*.png         (modo completo)
  geral/*.png                         (modo completo)
  entradas/*.png                      (exemplos da entrada da rede, se o modo usa máscara)
  resumo_rer.csv                      (tabela RER por fold / média por CV / total — formato antigo)
  resumo_metricas.csv                 (todas as métricas, formato longo, bom para pandas)
  summary.json                        (lido pelo compare.py)
"""
from __future__ import annotations

import csv
import json
import os

import torch

from leaf_measures.data import MEASURE_ORDER, Fold, build_dataset
from leaf_measures.evaluation import plots
from leaf_measures.evaluation.inference import run_inference
from leaf_measures.evaluation.metrics import METRIC_KEYS, compute_metrics, weighted_average
from leaf_measures.models import InputBuilder, build_model_and_inputs
from leaf_measures.training.checkpoint import load_weights
from leaf_measures.training.run_log import format_metrics
from leaf_measures.training.trainer import fold_paths, make_loader


def results_dir(cfg) -> str:
    return os.path.join(cfg.eval.results_dir, cfg.experiment.name)


def evaluate_fold(cfg, fold: Fold, device, full: bool = True) -> dict | None:
    ckpt_path = fold_paths(cfg, fold)["best"]
    test_csv = fold.csv("test")
    if not os.path.isfile(test_csv):
        print(f"[AVISO] {fold.id}: split de teste não encontrado ({test_csv}). Ignorando.")
        return None
    if not os.path.isfile(ckpt_path):
        print(f"[AVISO] {fold.id}: checkpoint não encontrado ({ckpt_path}). Ignorando.")
        return None

    model, inputs = build_model_and_inputs(cfg, num_outputs=len(MEASURE_ORDER), pretrained=False)
    model = model.to(device)
    try:
        ckpt = load_weights(ckpt_path, model, device)
    except Exception as exc:  # noqa: BLE001
        print(f"[AVISO] {fold.id}: falha ao carregar {ckpt_path} ({exc}). Ignorando.")
        return None

    ds = build_dataset(cfg, fold, "test", load_masks=inputs.needs_masks)
    loader = make_loader(ds, cfg, shuffle=False, batch_size=cfg.eval.batch_size)

    print(f"\n>>> {cfg.experiment.name} — {fold.id} | checkpoint iter {ckpt.get('iteration')} | "
          f"{len(ds)} imagens de teste")
    res = run_inference(model, loader, inputs, device)
    metrics = compute_metrics(res["preds"], res["targets"])
    print(format_metrics(metrics))

    out_dir = os.path.join(results_dir(cfg), fold.id)
    preds_np = {n: v.numpy() for n, v in res["preds"].items()}
    targets_np = {n: v.numpy() for n, v in res["targets"].items()}
    save_predictions_csv(res["names"], preds_np, targets_np, os.path.join(out_dir, "predicoes"))
    if full:
        plots.fold_plots(cfg.experiment.name, metrics, preds_np, targets_np, os.path.join(out_dir, "graficos"))

    return {
        "fold": fold.id,
        "cv": fold.cv,
        "split": fold.split,
        "n_samples": len(ds),
        "iteration": ckpt.get("iteration"),
        "metrics": metrics,
    }


def evaluate_experiment(cfg, folds: list[Fold], device, full: bool = True) -> dict:
    out_root = results_dir(cfg)
    os.makedirs(out_root, exist_ok=True)

    results = [r for r in (evaluate_fold(cfg, f, device, full) for f in folds) if r is not None]
    if not results:
        print(f"\nNenhum checkpoint encontrado para '{cfg.experiment.name}'. Nada a resumir.")
        return {}

    by_cv: dict = {}
    for r in results:
        by_cv.setdefault(r["cv"], []).append(r)
    cv_averages = {
        cv: {"n_samples": sum(e["n_samples"] for e in es), "n_folds": len(es), "metrics": weighted_average(es)}
        for cv, es in sorted(by_cv.items())
    }
    overall = {"n_samples": sum(r["n_samples"] for r in results), "metrics": weighted_average(results)}

    if full:
        plots.summary_plots(cfg.experiment.name, cv_averages, overall, out_root)
        _save_input_examples(cfg, folds, results, device, out_root)

    summary = {
        "experiment": cfg.experiment.name,
        "label": cfg.experiment.get("label") or cfg.experiment.name,
        "backbone": cfg.model.backbone,
        "input_mode": cfg.input.mode,
        "folds": results,
        "cv_averages": {str(k): v for k, v in cv_averages.items()},
        "overall": overall,
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_rer_table(summary, os.path.join(out_root, "resumo_rer.csv"))
    write_long_metrics(summary, os.path.join(out_root, "resumo_metricas.csv"))

    print("\n" + "#" * 60)
    print(f"RESUMO — {cfg.experiment.name}")
    for cv, avg in cv_averages.items():
        print(f"\n--- CV{cv} (média ponderada, {avg['n_folds']} fold(s), {avg['n_samples']} amostras) ---")
        print(format_metrics(avg["metrics"]))
    print(f"\n--- MÉDIA PONDERADA TOTAL ({overall['n_samples']} amostras) ---")
    print(format_metrics(overall["metrics"]))
    print("#" * 60)
    print(f"Resultados em: {out_root}")
    return summary


# ----------------------------------------------------------------- relatórios
def save_predictions_csv(names, preds_np, targets_np, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for n in MEASURE_ORDER:
        with open(os.path.join(save_dir, f"{n}_predictions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "predicted", "real"])
            for s, p, r in zip(names, preds_np[n], targets_np[n]):
                w.writerow([s, float(p), float(r)])


def write_rer_table(summary: dict, path: str):
    """Linhas = medidas; colunas = cada fold, média por CV e média total (RER %)."""
    folds_by_cv: dict = {}
    for r in summary["folds"]:
        folds_by_cv.setdefault(r["cv"], []).append(r)

    columns = []
    for cv in sorted(folds_by_cv):
        for r in sorted(folds_by_cv[cv], key=lambda r: r["split"]):
            columns.append((f"cv{cv} fold{r['split']}", r["metrics"]))
        columns.append((f"cv{cv} weighted av", summary["cv_averages"][str(cv)]["metrics"]))
    columns.append(("TOTAL weighted av", summary["overall"]["metrics"]))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([summary["label"]] + [c[0] for c in columns])
        for n in MEASURE_ORDER:
            w.writerow([n.capitalize()] + [f"{m[n]['rer']:.2f}%" for _, m in columns])
    print(f"Planilha de RER salva em: {path}")


def write_long_metrics(summary: dict, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "scope", "fold", "n_samples", "measure", *METRIC_KEYS])
        exp = summary["experiment"]
        for r in summary["folds"]:
            for n in MEASURE_ORDER:
                w.writerow([exp, "fold", r["fold"], r["n_samples"], n, *[r["metrics"][n][k] for k in METRIC_KEYS]])
        for cv, avg in summary["cv_averages"].items():
            for n in MEASURE_ORDER:
                w.writerow([exp, "cv_avg", f"cv{cv}", avg["n_samples"], n, *[avg["metrics"][n][k] for k in METRIC_KEYS]])
        ov = summary["overall"]
        for n in MEASURE_ORDER:
            w.writerow([exp, "total_avg", "", ov["n_samples"], n, *[ov["metrics"][n][k] for k in METRIC_KEYS]])


def _save_input_examples(cfg, folds, results, device, out_root, n: int = 4):
    """Para modos com máscara, salva exemplos do que a rede realmente recebe."""
    inputs = InputBuilder.from_config(cfg)
    if not inputs.needs_masks:
        return
    fold = next(f for f in folds if f.id == results[0]["fold"])
    ds = build_dataset(cfg, fold, "test", load_masks=True)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=min(n, len(ds)), shuffle=True, num_workers=0)))
    x = inputs(batch, torch.device("cpu"))
    plots.save_input_examples(batch["image"], x, os.path.join(out_root, "entradas"), title=cfg.input.mode)
