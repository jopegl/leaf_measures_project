import argparse

import torch

from common.datasets import LSIDBeansMeasuresDataset, LSIDBeansDataset
from common.metrics import calculate_measure_metrics, calculate_miou
from common.normalization import build_target_tensor, denormalize_columns, MEASURE_ORDER


def evaluate_resnet():
    from models.resnet_regressor import ResNetRegressor
    from configs import resnet_config as cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = LSIDBeansMeasuresDataset(root_dir=cfg.ROOT, split_file=cfg.TEST_CSV, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )

    model = ResNetRegressor(
        backbone_name=cfg.BACKBONE, num_outputs=cfg.NUM_OUTPUTS, pretrained=False
    ).to(device)

    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint carregado: {cfg.CHECKPOINT_PATH} (iteração {checkpoint.get('iteration')})")
    print(f"Imagens de teste: {len(test_dataset)}")

    preds_by_measure = {name: [] for name in MEASURE_ORDER}
    targets_by_measure = {name: [] for name in MEASURE_ORDER}

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)
            out = model(images)

            out_denorm = denormalize_columns(out.detach().cpu())
            target_denorm = denormalize_columns(target.detach().cpu())
            for name in MEASURE_ORDER:
                preds_by_measure[name].append(out_denorm[name])
                targets_by_measure[name].append(target_denorm[name])

    print("\n" + "=" * 60)
    print("MÉTRICAS FINAIS NO TESTE (ResNetRegressor)")
    for name in MEASURE_ORDER:
        preds = torch.cat(preds_by_measure[name])
        targets = torch.cat(targets_by_measure[name])
        mae, rmse, rer, r2 = calculate_measure_metrics(preds, targets)
        print(f"  {name.capitalize():<10} MAE={mae:.4f}  RMSE={rmse:.4f}  RER={rer:.2f}%  R²={r2:.4f}")
    print("=" * 60)


def evaluate_deeplab():
    from models.deeplab_just_measures import DeepLab
    from configs import deeplab_config as cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = LSIDBeansDataset(root_dir=cfg.ROOT, split_file=cfg.TEST_CSV, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )

    model = DeepLab(
        num_classes=cfg.NUM_CLASSES, backbone=cfg.BACKBONE,
        output_stride=cfg.OUTPUT_STRIDE, sync_bn=cfg.SYNC_BN, freeze_bn=cfg.FREEZE_BN,
    ).to(device)

    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint carregado: {cfg.CHECKPOINT_PATH} (iteração {checkpoint.get('iteration')})")
    print(f"Imagens de teste: {len(test_dataset)}")

    preds_by_measure = {name: [] for name in MEASURE_ORDER}
    targets_by_measure = {name: [] for name in MEASURE_ORDER}
    miou_total = 0.0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            seg_mask = batch["seg_mask"].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)

            seg_out, measures_out = model(images)
            miou_total += calculate_miou(seg_out, seg_mask, cfg.NUM_CLASSES)

            out_denorm = denormalize_columns(measures_out.detach().cpu())
            target_denorm = denormalize_columns(target.detach().cpu())
            for name in MEASURE_ORDER:
                preds_by_measure[name].append(out_denorm[name])
                targets_by_measure[name].append(target_denorm[name])

    print("\n" + "=" * 60)
    print("MÉTRICAS FINAIS NO TESTE (DeepLab multi-tarefa)")
    print(f"  mIoU: {miou_total / len(test_loader):.4f}\n")
    for name in MEASURE_ORDER:
        preds = torch.cat(preds_by_measure[name])
        targets = torch.cat(targets_by_measure[name])
        mae, rmse, rer, r2 = calculate_measure_metrics(preds, targets)
        print(f"  {name.capitalize():<10} MAE={mae:.4f}  RMSE={rmse:.4f}  RER={rer:.2f}%  R²={r2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "deeplab"], required=True)
    args = parser.parse_args()

    if args.model == "resnet":
        evaluate_resnet()
    else:
        evaluate_deeplab()
