"""
Compara experimentos já avaliados (lê results/<experimento>/summary.json).

    python compare.py resnet50_rgb resnet50_hadamard resnet50_mask
    python compare.py resnet50_rgb efficientnet_b0_rgb --metric rer --out results/comparacoes/backbones.png
"""
import argparse
import csv
import json
import os

from leaf_measures.data.normalization import MEASURE_ORDER
from leaf_measures.evaluation.metrics import METRIC_KEYS
from leaf_measures.evaluation.plots import plot_model_comparison


def main():
    p = argparse.ArgumentParser(description="Comparação entre experimentos avaliados")
    p.add_argument("experiments", nargs="+", help="Nomes dos experimentos (pastas em results/)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default=None, help="PNG de saída (padrão: results/comparacoes/<a>_vs_<b>.png)")
    args = p.parse_args()

    summaries = {}
    for exp in args.experiments:
        path = os.path.join(args.results_dir, exp, "summary.json")
        if not os.path.isfile(path):
            print(f"[AVISO] {path} não encontrado — rode evaluate.py para '{exp}' antes. Ignorando.")
            continue
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        summaries[s.get("label") or exp] = s["overall"]["metrics"]

    if not summaries:
        print("Nada para comparar.")
        return

    print(f"\n{'experimento':<28}{'medida':<11}" + "".join(f"{k.upper():>10}" for k in METRIC_KEYS))
    for lab, m in summaries.items():
        for n in MEASURE_ORDER:
            print(f"{lab:<28}{n:<11}" + "".join(f"{m[n][k]:>10.4f}" for k in METRIC_KEYS))

    name = "_vs_".join(args.experiments)
    out = args.out or os.path.join(args.results_dir, "comparacoes", f"{name}.png")
    if len(summaries) >= 2:
        plot_model_comparison(summaries, out)
        print(f"\nGráfico salvo em: {out}")
    csv_path = os.path.splitext(out)[0] + ".csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "measure", *METRIC_KEYS])
        for lab, m in summaries.items():
            for n in MEASURE_ORDER:
                w.writerow([lab, n, *[m[n][k] for k in METRIC_KEYS]])
    print(f"Tabela salva em: {csv_path}")


if __name__ == "__main__":
    main()
