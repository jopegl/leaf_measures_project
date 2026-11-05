"""
Modelos e modos de entrada.

Contrato de um modelo registrado em MODELS:
  - classmethod from_config(model_cfg, in_channels, num_outputs)
  - forward(x) -> tensor [B, num_outputs] (medidas normalizadas)
  - backbone_params(), head_params(), set_backbone_trainable(bool)
"""
# Importar os módulos registra os modelos / modos no registry.
from leaf_measures.models import inputs, regressor  # noqa: F401
from leaf_measures.models.inputs import InputBuilder
from leaf_measures.registry import INPUT_MODES, MODELS


def build_model(cfg, in_channels: int, num_outputs: int):
    model_cls = MODELS.get(cfg.model.name)
    return model_cls.from_config(cfg.model, in_channels=in_channels, num_outputs=num_outputs)


def build_model_and_inputs(cfg, num_outputs: int, pretrained: bool | None = None):
    """Atalho: cria o InputBuilder e o modelo com o nº de canais certo para o modo."""
    builder = InputBuilder.from_config(cfg)
    if pretrained is not None:
        cfg = _with_pretrained(cfg, pretrained)
    return build_model(cfg, builder.in_channels, num_outputs), builder


def _with_pretrained(cfg, pretrained: bool):
    import copy

    cfg = copy.deepcopy(cfg)
    cfg.model.pretrained = pretrained
    return cfg


__all__ = ["build_model", "build_model_and_inputs", "InputBuilder", "MODELS", "INPUT_MODES"]
