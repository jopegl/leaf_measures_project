"""
Backbone (torchvision) + head MLP de regressão das 4 medidas.

Os nomes dos submódulos (`backbone.*`, `head.*`) e o head padrão
(Linear -> ReLU -> Dropout -> Linear, hidden_dims=[256], dropout=0.3) são os
mesmos do antigo ResNetRegressor e do ResNetMaskRegressor, então checkpoints
antigos de train_resnet_measures.py / train_resnet_hadamard.py (input.mode=rgb
ou hadamard) e de train_resnet_mask.py (input.mode=mask_concat) carregam direto.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from leaf_measures.models.backbones import build_backbone
from leaf_measures.registry import MODELS


def build_mlp_head(in_features: int, hidden_dims, num_outputs: int, dropout: float) -> nn.Sequential:
    layers = []
    prev = in_features
    for h in hidden_dims or []:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, num_outputs))
    return nn.Sequential(*layers)


@MODELS.register("regressor")
class Regressor(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 3,
        num_outputs: int = 4,
        head_hidden_dims=(256,),
        head_dropout: float = 0.3,
        extra_channels_init: str = "zeros",
    ):
        super().__init__()
        self.backbone, feat_dim = build_backbone(
            backbone, pretrained=pretrained, in_channels=in_channels, extra_channels_init=extra_channels_init
        )
        self.head = build_mlp_head(feat_dim, head_hidden_dims, num_outputs, head_dropout)

    @classmethod
    def from_config(cls, model_cfg, in_channels: int, num_outputs: int) -> "Regressor":
        head = model_cfg.get("head", {}) or {}
        return cls(
            backbone=model_cfg.backbone,
            pretrained=bool(model_cfg.get("pretrained", True)),
            in_channels=in_channels,
            num_outputs=num_outputs,
            head_hidden_dims=head.get("hidden_dims", [256]),
            head_dropout=float(head.get("dropout", 0.3)),
            extra_channels_init=model_cfg.get("extra_channels_init", "zeros"),
        )

    def forward(self, x):
        feats = torch.flatten(self.backbone(x), 1)
        return self.head(feats)

    # --- interface usada pelo trainer (todo modelo registrado deve expor) ---
    def backbone_params(self):
        return self.backbone.parameters()

    def head_params(self):
        return self.head.parameters()

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable
