"""
Treino de um fold. Mantém o comportamento dos antigos train_resnet_*.py:
  - AdamW com dois grupos (head e backbone, LRs diferentes) + OneCycleLR
  - backbone congelado nas primeiras `freeze_backbone_validations` validações
  - loss ponderada por medida (huber/mse), clip de gradiente
  - validação a cada `val_interval` iterações; salva o melhor por val loss

Novidades: last.pth para retomar sem perder progresso, seed por fold,
AMP opcional, snapshot da config do experimento em runs/<exp>/config.yaml.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from leaf_measures.config import save_config
from leaf_measures.data import MEASURE_ORDER, Fold, build_dataset, build_target_tensor
from leaf_measures.evaluation.inference import run_inference
from leaf_measures.evaluation.metrics import compute_metrics
from leaf_measures.models import build_model_and_inputs
from leaf_measures.training.checkpoint import resume_if_possible, save_checkpoint
from leaf_measures.training.losses import weighted_measures_loss
from leaf_measures.training.run_log import append_log_row, init_log_csv, now, print_val_block


def run_dir(cfg) -> str:
    return os.path.join(cfg.experiment.output_dir, cfg.experiment.name)


def fold_paths(cfg, fold: Fold) -> dict:
    d = os.path.join(run_dir(cfg), fold.id)
    return {"dir": d, "best": os.path.join(d, "best.pth"), "last": os.path.join(d, "last.pth")}


def model_signature(cfg) -> dict:
    """O que precisa bater para um checkpoint ser retomado com esta config."""
    return {"model": cfg.model.name, "backbone": cfg.model.backbone, "input_mode": cfg.input.mode}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, cfg, shuffle: bool, batch_size: int | None = None):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=bool(cfg.train.pin_memory),
        persistent_workers=cfg.train.num_workers > 0,
    )


def make_loss_fn(cfg, device):
    weights = torch.tensor(cfg.loss.weights, dtype=torch.float32, device=device)
    if len(weights) != len(MEASURE_ORDER):
        raise ValueError(f"loss.weights precisa ter {len(MEASURE_ORDER)} valores ({MEASURE_ORDER})")
    return partial(
        weighted_measures_loss,
        weights=weights,
        loss_type=cfg.loss.type,
        huber_beta=float(cfg.loss.huber_beta),
    )


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)  # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass
class FoldResult:
    fold_id: str
    best_val_loss: float
    checkpoint: str


def train_fold(cfg, fold: Fold, device) -> FoldResult:
    t = cfg.train
    for key in ("iters", "val_interval", "log_every", "batch_size", "num_workers"):
        t[key] = int(t[key])
    paths = fold_paths(cfg, fold)
    log_csv = os.path.join(run_dir(cfg), "training_log.csv")
    set_seed(int(cfg.experiment.seed) + fold.cv * 100 + fold.split)

    print("\n" + "#" * 70)
    print(f"### {cfg.experiment.name} | FOLD {fold.id}")
    print(f"### train={fold.csv('train')}  val={fold.csv('val')}")
    print(f"### checkpoints -> {paths['dir']}")
    print("#" * 70)

    model, inputs = build_model_and_inputs(cfg, num_outputs=len(MEASURE_ORDER))
    model = model.to(device)

    train_ds = build_dataset(cfg, fold, "train", load_masks=inputs.needs_masks)
    val_ds = build_dataset(cfg, fold, "val", load_masks=inputs.needs_masks)
    print(f"Imagens de treino: {len(train_ds)} | validação: {len(val_ds)} | entrada: {cfg.input.mode} "
          f"({inputs.in_channels} canais) | backbone: {cfg.model.backbone}")
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.head_params(), "lr": t.lr_head},
            {"params": model.backbone_params(), "lr": t.lr_backbone},
        ],
        weight_decay=t.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[t.lr_head, t.lr_backbone], total_steps=t.iters, pct_start=t.pct_start
    )
    use_amp = bool(t.amp) and device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)
    loss_fn = make_loss_fn(cfg, device)

    signature = model_signature(cfg)
    ckpt = resume_if_possible([paths["last"], paths["best"]], device, model, optimizer, scheduler, scaler,
                              expected_signature=signature)
    if ckpt and scheduler.total_steps != t.iters:
        raise RuntimeError(
            f"O checkpoint de {fold.id} foi treinado com train.iters={scheduler.total_steps}, "
            f"mas a config pede {t.iters}. O OneCycleLR não pode ser estendido: use outro "
            "experiment.name ou mantenha o mesmo nº de iterações."
        )
    iteration = ckpt.get("iteration", 0)
    best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
    unfrozen = ckpt.get("unfrozen", False)
    model.set_backbone_trainable(unfrozen)
    unfreeze_at = int(t.freeze_backbone_validations) * int(t.val_interval)

    if iteration >= t.iters:
        print(f"Fold {fold.id} já treinado ({iteration}/{t.iters}). Pulando.")
        return FoldResult(fold.id, best_val_loss, paths["best"])

    print(f"Começando fold {fold.id} [{now()}]")
    while iteration < t.iters:
        model.train()
        for batch in train_loader:
            if iteration >= t.iters:
                break

            x = inputs(batch, device)
            target = build_target_tensor(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
            out = out.float()
            if not torch.isfinite(out).all():
                print(f"[{fold.id} | iter {iteration}] saída com NaN/Inf! "
                      f"min={out.min().item()} max={out.max().item()}")

            loss, per_comp = loss_fn(out, target)
            scaler.scale(loss).backward()
            if t.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(t.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            iteration += 1

            if iteration % t.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                comps = " ".join(f"{n[0].upper()}={v:.4f}" for n, v in zip(MEASURE_ORDER, per_comp.tolist()))
                print(f"[{fold.id} | {now()}] Iter [{iteration}/{t.iters}] LR: {lr:.8f} "
                      f"Loss: {loss.item():.4f} ({comps})")

            if iteration % t.val_interval == 0:
                if not unfrozen and iteration >= unfreeze_at:
                    print(f">>> [{fold.id}] Liberando fine-tuning do backbone (unfreeze).")
                    model.set_backbone_trainable(True)
                    unfrozen = True

                res = run_inference(model, val_loader, inputs, device, loss_fn=loss_fn, amp=use_amp)
                metrics = compute_metrics(res["preds"], res["targets"])
                lr = optimizer.param_groups[0]["lr"]
                print_val_block(f"Val {fold.id}", res["loss"], metrics, iteration, t.iters, lr)
                append_log_row(log_csv, fold.id, iteration, lr, res["loss"], metrics)

                improved = res["loss"] < best_val_loss
                if improved:
                    best_val_loss = res["loss"]

                state = dict(
                    iteration=iteration,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    scaler_state_dict=scaler.state_dict() if use_amp else None,
                    val_loss=res["loss"],
                    best_val_loss=best_val_loss,
                    unfrozen=unfrozen,
                    experiment=cfg.experiment.name,
                    fold=fold.id,
                    signature=signature,
                )
                if improved:
                    save_checkpoint(paths["best"], **state)
                    print(f">>> [{fold.id}] Melhor modelo salvo em {paths['best']}")
                save_checkpoint(paths["last"], **state)
                model.train()

    print(f"\nFold {fold.id} finalizado. Melhor val loss: {best_val_loss:.4f}")
    return FoldResult(fold.id, best_val_loss, paths["best"])


def train_experiment(cfg, folds: list[Fold], device) -> list[FoldResult]:
    os.makedirs(run_dir(cfg), exist_ok=True)
    save_config(cfg, os.path.join(run_dir(cfg), "config.yaml"))
    init_log_csv(os.path.join(run_dir(cfg), "training_log.csv"))

    results = [train_fold(cfg, f, device) for f in folds]

    print("\n" + "=" * 70)
    print(f"RESUMO DO TREINO — {cfg.experiment.name}")
    print("=" * 70)
    for r in results:
        print(f"{r.fold_id}: best_val_loss={r.best_val_loss:.4f} -> {r.checkpoint}")
    if results:
        avg = sum(r.best_val_loss for r in results) / len(results)
        print(f"\nMédia do best_val_loss em {len(results)} fold(s): {avg:.4f}")
    print(f"Avalie no teste com: python evaluate.py --config <mesmo yaml>")
    return results
