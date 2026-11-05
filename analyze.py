"""
Análise de erros: piores/melhores imagens e indícios de padrões.

Usa as predições salvas pelo evaluate.py: rode-o antes (pode ser com --light).

    python analyze.py resnet50_rgb
    python analyze.py resnet50_rgb efficientnet_b3_rgb        # + comparação entre modelos
    python analyze.py resnet50_rgb --folds cv1_split1 cv2_split2 --top-n 16 --quantile 0.2
    python analyze.py resnet50_rgb --no-images                 # só CSVs e indicios.md

Saída em results/analise/<experimento>/indicios.md (+ CSVs, grades de imagens, gráficos).
As características das imagens ficam em cache em results/analise/features.csv.
"""
import argparse
import os
import sys

from leaf_measures.analysis import analyze
from leaf_measures.config import load_config


def find_config(exp: str, runs_dir: str = "runs") -> str:
    for path in (os.path.join(runs_dir, exp, "config.yaml"),
                 os.path.join("configs", "experiments", f"{exp}.yaml"),
                 os.path.join("configs", "base.yaml")):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Nenhuma config encontrada para obter os caminhos do dataset.")


def main():
    p = argparse.ArgumentParser(description="Análise de erros por imagem e busca de padrões")
    p.add_argument("experiments", nargs="+", help="Experimentos já avaliados (pastas em results/)")
    p.add_argument("--folds", nargs="+", default=None, help="Só estes folds (ex.: cv1_split1 cv2_split2)")
    p.add_argument("--top-n", type=int, default=12, help="Quantas piores/melhores mostrar (padrão 12)")
    p.add_argument("--quantile", type=float, default=0.1,
                   help="Fração considerada 'piores' na busca de padrões (padrão 0.1 = 10%%)")
    p.add_argument("--no-images", action="store_true", help="Não gera as grades de imagens")
    p.add_argument("--recompute-features", action="store_true", help="Recalcula o cache de características")
    p.add_argument("--config", default=None, help="Config de onde ler data.root (padrão: a do 1º experimento)")
    p.add_argument("--results-dir", default=None, help="Padrão: eval.results_dir da config")
    args = p.parse_args()

    cfg = load_config(args.config or find_config(args.experiments[0]))
    analyze(
        args.experiments,
        cfg.data,
        results_dir=args.results_dir or cfg.eval.results_dir,
        folds=args.folds,
        top_n=args.top_n,
        quantile=args.quantile,
        images=not args.no_images,
        recompute_features=args.recompute_features,
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"\nERRO: {exc}")
        sys.exit(1)
