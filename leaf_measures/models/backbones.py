"""
Fábrica genérica de backbones do torchvision.

Qualquer nome aceito por `torchvision.models.get_model` funciona
(resnet18/34/50/101, resnext50_32x4d, wide_resnet50_2, efficientnet_b0..b7,
efficientnet_v2_s, convnext_tiny/small, regnet_y_*, mobilenet_v3_large,
densenet121, swin_t, vit_b_16 ...). Liste com
`python -c "import torchvision; print(torchvision.models.list_models())"`.

A camada de classificação é trocada por Identity e a dimensão das features é
lida do Linear removido. Se in_channels != 3, a primeira Conv2d é adaptada:
os pesos RGB pré-treinados são mantidos e os canais extras são inicializados
conforme `extra_channels_init`:
  - "zeros": pesos zerados (como no ResNetMaskRegressor original; no início a
             rede se comporta exatamente como a pré-treinada em RGB)
  - "mean":  média dos pesos RGB

Atenção: os ViT do torchvision exigem entrada 224x224 (image_size fixo).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

_HEAD_ATTRS = ("fc", "classifier", "heads", "head")


def build_backbone(
    name: str, pretrained: bool = True, in_channels: int = 3, extra_channels_init: str = "zeros"
) -> tuple[nn.Module, int]:
    """Retorna (backbone_sem_head, feat_dim)."""
    weights = tvm.get_model_weights(name).DEFAULT if pretrained else None
    net = tvm.get_model(name, weights=weights)

    feat_dim = _strip_classifier(net, name)
    if in_channels != 3:
        _adapt_first_conv(net, in_channels, extra_channels_init)
    return net, feat_dim


def _strip_classifier(net: nn.Module, name: str) -> int:
    for attr in _HEAD_ATTRS:
        head = getattr(net, attr, None)
        if head is None or not isinstance(head, nn.Module):
            continue
        if isinstance(head, nn.Linear):
            setattr(net, attr, nn.Identity())
            return head.in_features
        if isinstance(head, nn.Sequential):
            # Troca só o último Linear (mantém LayerNorm/Flatten do ConvNeXt, p.ex.)
            for idx in range(len(head) - 1, -1, -1):
                if isinstance(head[idx], nn.Linear):
                    in_features = head[idx].in_features
                    head[idx] = nn.Identity()
                    return in_features
    raise ValueError(
        f"Não sei remover a camada de classificação de '{name}'. "
        f"Adicione o caso em leaf_measures/models/backbones.py::_strip_classifier."
    )


def _adapt_first_conv(net: nn.Module, in_channels: int, extra_init: str = "zeros"):
    if extra_init not in ("zeros", "mean"):
        raise ValueError(f"extra_channels_init deve ser 'zeros' ou 'mean', não '{extra_init}'")
    for parent_name, parent in net.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                new = nn.Conv2d(
                    in_channels,
                    child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                )
                with torch.no_grad():
                    w = child.weight
                    new.weight[:, :3] = w
                    if in_channels > 3:
                        if extra_init == "mean":
                            new.weight[:, 3:] = w.mean(dim=1, keepdim=True)
                        else:
                            new.weight[:, 3:] = 0
                    if child.bias is not None:
                        new.bias.copy_(child.bias)
                setattr(parent, child_name, new)
                return
    raise ValueError("Nenhuma Conv2d com 3 canais de entrada encontrada para adaptar.")
