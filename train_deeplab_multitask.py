from datetime import datetime

import torch

from common.datasets import LSIDBeansDataset
from common.metrics import calculate_measure_metrics, calculate_miou
from common.losses import weighted_measures_loss
from common.normalization import build_target_tensor, denormalize_columns, MEASURE_ORDER
from common.checkpoint import save_checkpoint, load_checkpoint_if_exists
from configs import deeplab_config as cfg

# Mantém a mesma dependência externa do script original.
from models.deeplab_just_measures import DeepLab


def run_validation(model, val_loader, device, num_classes, measure_weights):
    model.eval()

    preds_by_measure = {name: [] for name in MEASURE_ORDER}
    targets_by_measure = {name: [] for name in MEASURE_ORDER}

    val_total_loss = 0.0
    val_seg_loss = 0.0
    val_measures_loss = 0.0
    val_miou = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            seg_mask = batch["seg_mask"].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)

            seg_out, measures_out = model(images)

            loss_seg = torch.nn.functional.cross_entropy(seg_out, seg_mask)
            loss_measures, _ = weighted_measures_loss(measures_out, target, measure_weights, loss_type="mse")

            loss = cfg.SEG_LOSS_WEIGHT * loss_seg + cfg.MEASURES_LOSS_WEIGHT * loss_measures

            val_total_loss += loss.item()
            val_seg_loss += loss_seg.item()
            val_measures_loss += loss_measures.item()
            val_miou += calculate_miou(seg_out, seg_mask, num_classes)

            out_denorm = denormalize_columns(measures_out.detach().cpu())
            target_denorm = denormalize_columns(target.detach().cpu())
            for name in MEASURE_ORDER:
                preds_by_measure[name].append(out_denorm[name])
                targets_by_measure[name].append(target_denorm[name])

    n = len(val_loader)
    val_total_loss /= n
    val_seg_loss /= n
    val_measures_loss /= n
    val_miou /= n

    metrics = {}
    for name in MEASURE_ORDER:
        preds = torch.cat(preds_by_measure[name])
        targets = torch.cat(targets_by_measure[name])
        metrics[name] = calculate_measure_metrics(preds, targets)

    return val_total_loss, val_seg_loss, val_measures_loss, val_miou, metrics


def print_metrics_block(title, val_total_loss, val_seg_loss, val_measures_loss, val_miou, metrics, iteration, iters, lr):
    print("\n" + "=" * 60)
    print(f"[{title}] Iteration [{iteration}/{iters}]")
    print(f"LR: {lr:.8f}")
    print(f"Loss: {val_total_loss:.4f}")
    print(f"  Segmentation: {val_seg_loss:.4f}")
    print(f"  Measures:     {val_measures_loss:.4f}")
    print(f"  mIoU:         {val_miou:.4f}\n")
    print(f"  {title} Measures:")
    for name in MEASURE_ORDER:
        mae, rmse, rer, r2 = metrics[name]
        print(f"    {name.capitalize():<10} MAE={mae:.4f}  RMSE={rmse:.4f}  RER={rer:.2f}%  R²={r2:.4f}")
    print("=" * 60)


def main():
    train_dataset = LSIDBeansDataset(root_dir=cfg.ROOT, split_file=cfg.TRAIN_CSV, augment=True)
    val_dataset = LSIDBeansDataset(root_dir=cfg.ROOT, split_file=cfg.VAL_CSV, augment=False)
    # test_dataset NÃO é criado aqui de propósito — ver evaluate_test.py.

    print(f"Imagens de treino:    {len(train_dataset)}")
    print(f"Imagens de validação: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLab(
        num_classes=cfg.NUM_CLASSES,
        backbone=cfg.BACKBONE,
        output_stride=cfg.OUTPUT_STRIDE,
        sync_bn=cfg.SYNC_BN,
        freeze_bn=cfg.FREEZE_BN,
    ).to(device)

    train_params = [
        {"params": model.get_1x_lr_params(), "lr": cfg.LEARNING_RATE},
        {"params": model.get_10x_lr_params(), "lr": cfg.LEARNING_RATE * 10},
    ]
    optimizer = torch.optim.SGD(
        train_params, lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=cfg.ITERS, power=cfg.POWER)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )

    measure_weights = torch.tensor(cfg.MEASURE_WEIGHTS_VALUES, device=device)

    iteration, best_val_loss, _ = load_checkpoint_if_exists(cfg.CHECKPOINT_PATH, device, model, optimizer, scheduler)

    print(f"Começando [{datetime.now().strftime('%H:%M:%S')}]")

    while iteration < cfg.ITERS:
        model.train()

        for batch in train_loader:
            if iteration >= cfg.ITERS:
                break

            images = batch["image"].to(device, non_blocking=True)
            seg_mask = batch["seg_mask"].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)

            optimizer.zero_grad()
            seg_out, measures_out = model(images)

            if torch.isnan(measures_out).any() or torch.isinf(measures_out).any():
                print(
                    f"[iter {iteration}] measures_out tem NaN/Inf! "
                    f"min={measures_out.min().item()}, max={measures_out.max().item()}"
                )

            loss_seg = torch.nn.functional.cross_entropy(seg_out, seg_mask)
            loss_measures, _ = weighted_measures_loss(measures_out, target, measure_weights, loss_type="mse")
            loss = cfg.SEG_LOSS_WEIGHT * loss_seg + cfg.MEASURES_LOSS_WEIGHT * loss_measures

            loss.backward()
            optimizer.step()
            scheduler.step()

            iteration += 1

            if iteration % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Iteration [{iteration}/{cfg.ITERS}] "
                    f"LR: {current_lr:.8f} Loss: {loss.item():.4f} "
                    f"Seg: {loss_seg.item():.4f} Measures: {loss_measures.item():.4f}"
                )

            if iteration % cfg.VAL_INTERVAL == 0:
                # >>> Seleção de checkpoint feita SEMPRE no val_loader <<<
                val_total_loss, val_seg_loss, val_measures_loss, val_miou, val_metrics = run_validation(
                    model, val_loader, device, cfg.NUM_CLASSES, measure_weights
                )
                current_lr = optimizer.param_groups[0]["lr"]
                print_metrics_block(
                    "Val", val_total_loss, val_seg_loss, val_measures_loss, val_miou,
                    val_metrics, iteration, cfg.ITERS, current_lr,
                )

                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    save_checkpoint(
                        cfg.CHECKPOINT_PATH,
                        iteration=iteration,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        val_loss=val_total_loss,
                        val_seg_loss=val_seg_loss,
                        val_measures_loss=val_measures_loss,
                        val_miou=val_miou,
                    )
                    print(">>> Melhor modelo salvo! (baseado em VAL, não em teste)")

                print()

    print("\nTreinamento finalizado.")
    print(f"Melhor validation loss: {best_val_loss:.4f}")
    print("Rode evaluate_test.py para obter a métrica final (imparcial) no conjunto de teste.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
