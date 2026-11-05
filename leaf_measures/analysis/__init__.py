"""
Análise de erros por imagem: melhores/piores, padrões e indícios.

Parte das predições já salvas pelo evaluate.py; não roda modelo nenhum.
Entrada de uso: analyze.py (raiz do repo). Organização:
  features.py  características de cada imagem (XML, imagem, máscara), com cache
  ranking.py   erro por imagem, piores/melhores, comparação entre modelos
  patterns.py  estatística e geração dos indícios
  report.py    CSVs, grades de imagens, gráficos e indicios.md
"""
from __future__ import annotations

import os

from leaf_measures.analysis.features import load_or_compute_features
from leaf_measures.analysis.patterns import find_clues
from leaf_measures.analysis.ranking import agreement, cross_model, load_predictions, per_image_errors
from leaf_measures.analysis.report import write_comparison_report, write_experiment_report


def analyze(experiments, data_cfg, results_dir="results", folds=None, top_n=12, quantile=0.1,
            images=True, recompute_features=False):
    out_root = os.path.join(results_dir, "analise")
    features = load_or_compute_features(
        os.path.join(out_root, "features.csv"), data_cfg.root, data_cfg.height, data_cfg.width,
        recompute=recompute_features,
    )

    tables = {}
    for exp in experiments:
        print(f"\n=== Analisando {exp} ===")
        per_img = per_image_errors(load_predictions(results_dir, exp, folds))
        missing = per_img.index.difference(features.index)
        if len(missing):
            print(f"[AVISO] {len(missing)} imagem(ns) das predições não encontradas no dataset "
                  f"(ex.: {list(missing[:3])}). Elas ficam sem características.")
        table = per_img.join(features, how="left")
        clues, stats = find_clues(table, quantile)
        out_dir = os.path.join(out_root, exp)
        write_experiment_report(exp, table, clues, stats, out_dir, top_n, quantile, images)
        tables[exp] = table
        _print_clues(clues, out_dir)

    if len(tables) > 1:
        cross = cross_model(tables, quantile)
        out_dir = os.path.join(out_root, "comparacao_" + "_vs_".join(tables))
        write_comparison_report(cross, agreement(tables), features, tables, out_dir, top_n, quantile, images)
        n_all = int((cross["classificacao"] == "difícil para todos").sum())
        print(f"\nComparação entre modelos: {n_all} imagem(ns) difíceis para todos -> {out_dir}")
    return tables


def _print_clues(clues, out_dir, limit=8):
    main = [c for c in clues if c.strength in ("forte", "moderado")]
    print(f"\n{len(main)} indício(s) forte(s)/moderado(s):")
    for c in main[:limit]:
        print(f"  [{c.strength:<8}] [{c.scope}] {c.text}")
    if len(main) > limit:
        print(f"  ... e mais {len(main) - limit}")
    print(f"Relatório completo: {os.path.join(out_dir, 'indicios.md')}")
