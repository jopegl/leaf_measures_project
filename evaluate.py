"""
Avalia no TESTE os checkpoints best.pth de um experimento.

    python evaluate.py --config configs/experiments/resnet50_rgb.yaml
    python evaluate.py -c configs/experiments/resnet50_rgb.yaml --light      # só CSVs/JSON, sem gráficos
    python evaluate.py -c configs/experiments/resnet50_rgb.yaml --folds cv1_split1

Folds sem checkpoint são ignorados (com aviso).
"""
import sys

from leaf_measures.cli import base_parser, resolve
from leaf_measures.evaluation.evaluator import evaluate_experiment


def main():
    p = base_parser("Avaliação no conjunto de teste")
    p.add_argument("--light", action="store_true", help="Não gera gráficos (só CSVs e summary.json)")
    args = p.parse_args()
    cfg, folds, device = resolve(args)
    evaluate_experiment(cfg, folds, device, full=not args.light)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(0)
