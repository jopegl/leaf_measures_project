"""
Modos de entrada: como transformar um batch do dataset no tensor que entra na rede.

Cada modo é uma função (image, masks) -> tensor, registrada com:
    in_channels  -> nº de canais que a rede recebe (adapta a 1ª conv se != 3)
    needs_masks  -> se o dataset precisa carregar as máscaras de segmentação

`image` chega em [0, 1] e, se data.imagenet_normalize=true, já normalizada
(mean/std ImageNet) antes de o modo ser aplicado. `masks` é [B, 2, H, W]
(canal 0 = folha, canal 1 = marcador) ou None.

Para criar um modo novo, basta adicionar uma função aqui:

    @INPUT_MODES.register("so_folha", in_channels=3, needs_masks=True)
    def so_folha(image, masks):
        return image * masks[:, :1]
"""
from __future__ import annotations

import torch

from leaf_measures.registry import INPUT_MODES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@INPUT_MODES.register("rgb", in_channels=3, needs_masks=False)
def rgb(image, masks):
    return image


@INPUT_MODES.register("hadamard", in_channels=3, needs_masks=True)
def hadamard(image, masks):
    """Produto de Hadamard: mantém só folha + marcador (união dos 2 canais)."""
    union = (masks > 0.5).any(dim=1, keepdim=True).to(image.dtype)
    return image * union


@INPUT_MODES.register("mask_concat", in_channels=5, needs_masks=True)
def mask_concat(image, masks):
    """RGB + 2 canais de máscara empilhados (entrada de 5 canais)."""
    return torch.cat([image, masks.to(image.dtype)], dim=1)


def mode_info(name: str) -> dict:
    return INPUT_MODES.meta(name)


def normalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


class InputBuilder:
    """Monta o tensor de entrada a partir do batch, conforme cfg.input.mode."""

    def __init__(self, mode: str, imagenet_normalize: bool = False):
        self.mode = mode
        self.fn = INPUT_MODES.get(mode)
        meta = INPUT_MODES.meta(mode)
        self.in_channels = meta["in_channels"]
        self.needs_masks = meta["needs_masks"]
        self.imagenet_normalize = imagenet_normalize

    @classmethod
    def from_config(cls, cfg) -> "InputBuilder":
        return cls(cfg.input.mode, bool(cfg.data.imagenet_normalize))

    def __call__(self, batch: dict, device) -> torch.Tensor:
        image = batch["image"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True) if self.needs_masks else None
        if self.imagenet_normalize:
            image = normalize_imagenet(image)
        return self.fn(image, masks)
