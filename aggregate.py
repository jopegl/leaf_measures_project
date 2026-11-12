"""
Novas formas de predição, com RER e demais métricas recalculados:

  # predição por FOLHA = média (ou mediana) das predições das imagens da folha
  python aggregate.py resnet50_rgb --leaf mean
  python aggregate.py resnet50_rgb --leaf median

  # ENSEMBLE: predição de cada imagem = média das predições de várias redes
  python aggregate.py resnet50_rgb efficientnet_b3_rgb resnet50_hadamard --ensemble mean

  # os dois: ensemble por imagem e depois agregação por folha
  python aggregate.py resnet50_rgb efficientnet_b3_rgb --ensemble mean --leaf mean

Usa as predições salvas pelo evaluate.py (rode-o antes para cada experimento).
O resultado vira um experimento em results/<nome>/ (nome automático ou --name),
no mesmo formato do evaluate.py: dá para usar com compare.py, analyze.py e o
Deep Viewer (deepviewer_test.csv em cada fold).
"""
import argparse
import sys

from leaf_measures.analysis.aggregation import AGG_FUNCS, LEAF_KEYS, aggregate
from leaf_measures.data.normalization import MEASURE_ORDER


def main():
    p = argparse.ArgumentParser(description="Predição por folha e/ou ensemble de redes")
    p.add_argument("experiments", nargs="+", help="Experimentos já avaliados (pastas em results/)")
    p.add_argument("--leaf", choices=list(AGG_FUNCS), default=None,
                   help="Agrega as imagens de cada folha: mean (média) ou median (mediana)")
    p.add_argument("--ensemble", choices=list(AGG_FUNCS), default=None,
                   help="Combina as redes por imagem: mean (média) ou median (mediana)")
    p.add_argument("--leaf-key", choices=LEAF_KEYS, default="pasta",
                   help="O que define uma folha: a pasta (padrão) ou pasta + medidas reais anotadas")
    p.add_argument("--folds", nargs="+", default=None, help="Só estes folds (ex.: cv1_split1 cv1_split2)")
    p.add_argument("--name", default=None, help="Nome do experimento gerado (padrão: automático)")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    try:
        s = aggregate(args.experiments, args.results_dir, ensemble_how=args.ensemble, leaf_how=args.leaf,
                      leaf_key=args.leaf_key, folds=args.folds, name=args.name)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERRO: {exc}")
        sys.exit(1)

    for w in s["warnings"]:
        print(f"[AVISO] {w}")
    comp = s["comparison"]
    print(f"\n{s['label']}  ->  {args.results_dir}/{s['experiment']}/")
    print(f"{len(s['folds'])} fold(s); média ponderada pelo nº de amostras de cada fold.\n")
    w = max(14, max(len(str(x)) for x in comp["predicao"]))
    print(f"{'RER total (%)':<{w}} {'n':>6} " + "".join(f"{m:>11}" for m in MEASURE_ORDER) + f"{'média':>9}")
    for _, r in comp.iterrows():
        print(f"{str(r['predicao']):<{w}} {int(r['n_amostras']):>6} "
              + "".join(f"{r[f'rer_{m}']:>11.2f}" for m in MEASURE_ORDER) + f"{r['rer_medio']:>9.2f}")
    print(f"\nTabela: {args.results_dir}/{s['experiment']}/comparacao_rer.csv")
    print(f"Comparar com outros: python compare.py {s['experiment']} <outros experimentos>")
    print(f"Piores casos:        python analyze.py {s['experiment']}")


if __name__ == "__main__":
    main()
