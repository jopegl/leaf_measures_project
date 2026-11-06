from __future__ import annotations

import argparse

import torch

from leaf_measures.config import load_config
from leaf_measures.data import folds_from_config


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", "-c", required=True, help="YAML do experimento (configs/experiments/...)")
    p.add_argument("--folds", nargs="+", default=None,
                   help="Só estes folds, ex.: --folds cv1_split1 cv2_split3 (padrão: folds da config)")
    p.add_argument("--start-from", default=None, metavar="FOLD",
                   help="Começa deste fold e segue a ordem da config, ex.: --start-from cv1_split5")
    p.add_argument("--set", nargs="+", default=[], metavar="CHAVE=VALOR",
                   help="Sobrescreve campos da config, ex.: --set model.backbone=resnet18 train.iters=2000")
    p.add_argument("--device", default=None, help="cuda | cpu | cuda:1 (padrão: cuda se disponível)")
    return p


def resolve(args):
    cfg = load_config(args.config, args.set)
    folds = folds_from_config(cfg, args.folds)
    if args.start_from:
        ids = [f.id for f in folds]
        if args.start_from not in ids:
            raise SystemExit(f"--start-from {args.start_from}: fold não está na lista ({ids[0]} ... {ids[-1]})")
        folds = folds[ids.index(args.start_from):]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Experimento: {cfg.experiment.name} | device: {device} | {len(folds)} fold(s)")
    return cfg, folds, device