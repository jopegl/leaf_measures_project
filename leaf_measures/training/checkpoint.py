"""
Checkpoints por fold:
  runs/<experimento>/<fold>/best.pth  -> melhor val loss (usado na avaliação)
  runs/<experimento>/<fold>/last.pth  -> estado da última validação (usado para retomar)

Nunca guardamos métricas de teste aqui: o treino é guiado só pela validação.
"""
from __future__ import annotations

import os

import torch


def save_checkpoint(path: str, **state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # escrita atômica: não corrompe se o treino cair no meio


def load_weights(path: str, model, device) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def check_signature(ckpt: dict, expected: dict | None, path: str):
    """Checkpoints antigos não têm assinatura: nesses casos, não há o que checar."""
    found = ckpt.get("signature")
    if expected and found and found != expected:
        raise RuntimeError(
            f"O checkpoint {path} é de outra configuração ({found}), e a atual é {expected}. "
            "Use outro experiment.name (ex.: --set experiment.name=novo_nome) ou apague a pasta do fold."
        )


def resume_if_possible(paths, device, model, optimizer=None, scheduler=None, scaler=None,
                       expected_signature: dict | None = None) -> dict:
    """
    Tenta retomar do primeiro checkpoint existente em `paths` (ex.: [last, best]).
    Retorna o dict do checkpoint (vazio se não houver nenhum).
    """
    for path in paths:
        if not os.path.exists(path):
            continue
        print(f"Retomando de: {path}")
        ckpt = torch.load(path, map_location=device)
        check_signature(ckpt, expected_signature, path)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"  iteração {ckpt.get('iteration', 0)} | melhor val loss "
              f"{ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf'))):.4f}")
        return ckpt
    print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
    return {}
