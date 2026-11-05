from leaf_measures.data.datasets import LSIDBeansDataset
from leaf_measures.data.folds import Fold, discover_folds, folds_from_config, parse_fold_id
from leaf_measures.data.normalization import (
    MEASURE_ORDER,
    MEASURE_RANGES,
    build_target_tensor,
    denormalize_columns,
)


def build_dataset(cfg, fold: Fold, subset: str, load_masks: bool) -> LSIDBeansDataset:
    """subset: 'train' | 'val' | 'test'. Augmentação só no treino (se data.augment=true)."""
    return LSIDBeansDataset(
        root_dir=cfg.data.root,
        split_file=fold.csv(subset),
        load_masks=load_masks,
        augment=bool(cfg.data.augment) and subset == "train",
        height=cfg.data.height,
        width=cfg.data.width,
    )
