"""
Treina um experimento em todos os folds definidos na config.

    python train.py --config configs/experiments/resnet50_rgb.yaml
    python train.py -c configs/experiments/resnet50_hadamard.yaml --folds cv1_split1
    python train.py -c configs/experiments/resnet50_rgb.yaml --set model.backbone=resnet18 experiment.name=resnet18_rgb

Retoma automaticamente de runs/<experimento>/<fold>/last.pth se existir.
"""
from leaf_measures.cli import base_parser, resolve
from leaf_measures.training.trainer import train_experiment


def main():
    args = base_parser("Treino de regressão das medidas da folha").parse_args()
    cfg, folds, device = resolve(args)
    train_experiment(cfg, folds, device)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
