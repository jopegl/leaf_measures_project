"""Inferência compartilhada entre a validação (treino) e a avaliação no teste."""
from __future__ import annotations

import torch

from leaf_measures.data.normalization import MEASURE_ORDER, build_target_tensor, denormalize_columns


@torch.no_grad()
def run_inference(model, loader, input_builder, device, loss_fn=None, amp: bool = False) -> dict:
    """
    Retorna:
      preds / targets : {medida: tensor 1D desnormalizado (cpu)}
      names           : ['<pasta_folha>/<imagem>', ...]
      loss            : loss média por batch (se loss_fn for dado), senão None
      loss_parts      : {medida: loss média daquela componente} (idem)
    """
    model.eval()
    preds = {n: [] for n in MEASURE_ORDER}
    targets = {n: [] for n in MEASURE_ORDER}
    names = []
    total_loss, n_batches = 0.0, 0
    total_parts = None

    for batch in loader:
        x = input_builder(batch, device)
        target = build_target_tensor(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
        out = out.float()
        if loss_fn is not None:
            loss, parts = loss_fn(out, target)
            total_loss += loss.item()
            total_parts = parts.cpu() if total_parts is None else total_parts + parts.cpu()
        n_batches += 1

        out_d = denormalize_columns(out.cpu())
        tgt_d = denormalize_columns(target.cpu())
        for n in MEASURE_ORDER:
            preds[n].append(out_d[n])
            targets[n].append(tgt_d[n])
        names.extend(f"{leaf}/{img}" for leaf, img in zip(batch["leaf_number"], batch["image_name"]))

    return {
        "preds": {n: torch.cat(v) for n, v in preds.items()},
        "targets": {n: torch.cat(v) for n, v in targets.items()},
        "names": names,
        "loss": (total_loss / max(n_batches, 1)) if loss_fn is not None else None,
        "loss_parts": ({m: float(v) / max(n_batches, 1) for m, v in zip(MEASURE_ORDER, total_parts.tolist())}
                       if total_parts is not None else None),
    }
