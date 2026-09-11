from datetime import datetime

import torch
import torch.nn as nn

from common.datasets import LSIDBeansMeasuresDataset
from common.metrics import calculate_measure_metrics
from common.losses import weighted_measures_loss
from common.normalization import build_target_tensor, denormalize_columns, MEASURE_ORDER
from common.checkpoint import save_checkpoint, load_checkpoint_if_exists
from models.resnet_regressor import ResNetRegressor
from configs import resnet_config as cfg


def run_validation(model, val_loader, device, measure_weights):
    model.eval()

    preds_by_measure = {name: [] for name in MEASURE_ORDER}
    targets_by_measure = {name: [] for name in MEASURE_ORDER}
    val_total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)

            out = model(images)
            loss, _ = weighted_measures_loss(
                out, target, measure_weights,
                loss_type=cfg.REGRESSION_LOSS, huber_beta=cfg.HUBER_BETA,
            )
            val_total_loss += loss.item()

            out_cpu = out.detach().cpu()
            target_cpu = target.detach().cpu()
            out_denorm = denormalize_columns(out_cpu)
            target_denorm = denormalize_columns(target_cpu)

            for name in MEASURE_ORDER:
                preds_by_measure[name].append(out_denorm[name])
                targets_by_measure[name].append(target_denorm[name])

    val_total_loss /= len(val_loader)

    metrics = {}
    for name in MEASURE_ORDER:
        preds = torch.cat(preds_by_measure[name])
        targets = torch.cat(targets_by_measure[name])
        metrics[name] = calculate_measure_metrics(preds, targets)

    return val_total_loss, metrics


def print_metrics_block(title, val_total_loss, metrics, iteration, iters, lr):
    print("\n" + "=" * 60)
    print(f"[{title}] Iteration [{iteration}/{iters}]")
    print(f"LR: {lr:.8f}")
    print(f"Loss: {val_total_loss:.4f}\n")
    print(f"  {title} Measures:")
    for name in MEASURE_ORDER:
        mae, rmse, rer, r2 = metrics[name]
        print(f"    {name.capitalize():<10} MAE={mae:.4f}  RMSE={rmse:.4f}  RER={rer:.2f}%  R²={r2:.4f}")
    print("=" * 60)


def main():
    train_dataset = LSIDBeansMeasuresDataset(root_dir=cfg.ROOT, split_file=cfg.TRAIN_CSV, augment=True)
    val_dataset = LSIDBeansMeasuresDataset(root_dir=cfg.ROOT, split_file=cfg.VAL_CSV, augment=False)
    # Note: test_dataset NÃO é criado aqui de propósito.

    print(f"Imagens de treino:     {len(train_dataset)}")
    print(f"Imagens de validação:  {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetRegressor(
        backbone_name=cfg.BACKBONE, num_outputs=cfg.NUM_OUTPUTS, pretrained=cfg.PRETRAINED
    ).to(device)

    model.set_backbone_trainable(False)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.head_params(), "lr": cfg.LR_HEAD},
            {"params": model.backbone_params(), "lr": cfg.LR_BACKBONE},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg.LR_HEAD, cfg.LR_BACKBONE],
        total_steps=cfg.ITERS,
        pct_start=0.1,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )

    measure_weights = torch.tensor(cfg.MEASURE_WEIGHTS_VALUES, device=device)

    iteration, best_val_loss, checkpoint = load_checkpoint_if_exists(
        cfg.CHECKPOINT_PATH, device, model, optimizer, scheduler
    )
    unfrozen = checkpoint.get("unfrozen", False)
    model.set_backbone_trainable(unfrozen)

    print(f"Começando [{datetime.now().strftime('%H:%M:%S')}]")

    val_count = 0

    while iteration < cfg.ITERS:
        model.train()

        for batch in train_loader:
            if iteration >= cfg.ITERS:
                break

            images = batch["image"].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)

            optimizer.zero_grad()
            out = model(images)

            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"[iter {iteration}] saída tem NaN/Inf! min={out.min().item()}, max={out.max().item()}")

            loss, per_component = weighted_measures_loss(
                out, target, measure_weights,
                loss_type=cfg.REGRESSION_LOSS, huber_beta=cfg.HUBER_BETA,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            iteration += 1

            if iteration % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Iteration [{iteration}/{cfg.ITERS}] "
                    f"LR: {current_lr:.8f} Loss: {loss.item():.4f} "
                    f"(L={per_component[0]:.4f} W={per_component[1]:.4f} "
                    f"P={per_component[2]:.4f} A={per_component[3]:.4f})"
                )

            if iteration % cfg.VAL_INTERVAL == 0:
                val_count += 1

                if not unfrozen and val_count >= cfg.FREEZE_BACKBONE_EPOCHS:
                    print(">>> Liberando fine-tuning do backbone (unfreeze).")
                    model.set_backbone_trainable(True)
                    unfrozen = True

                # >>> Seleção de checkpoint feita SEMPRE no val_loader <<<
                val_total_loss, val_metrics = run_validation(model, val_loader, device, measure_weights)
                current_lr = optimizer.param_groups[0]["lr"]
                print_metrics_block("Val", val_total_loss, val_metrics, iteration, cfg.ITERS, current_lr)

                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    save_checkpoint(
                        cfg.CHECKPOINT_PATH,
                        iteration=iteration,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        val_loss=val_total_loss,
                        unfrozen=unfrozen,
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
