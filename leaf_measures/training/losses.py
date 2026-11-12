"""
Perdas de regressão das 4 medidas, ponderadas por componente
(length, width, perimeter, area — ordem de MEASURE_ORDER).

loss.type no YAML:
  mse        erro quadrático nas medidas normalizadas (0–1)
  huber      Huber/SmoothL1 nas medidas normalizadas (padrão antigo)
  mse_log    MSLE: erro quadrático entre os LOGs das medidas reais (cm, cm²)
  huber_log  Huber entre os LOGs das medidas reais
  l1_log     erro absoluto entre os LOGs das medidas reais

As versões *_log medem erro RELATIVO, como o RER: log(pred) − log(real) ≈
erro percentual (0,05 ≈ 5%). Assim, errar 1,5 cm² numa folha de 15 cm² pesa
quase 3× mais que o mesmo erro numa de 40 cm², e erros para cima e para
baixo pesam igual (o que não acontece com |erro| ÷ real, o MAPE).
Nelas, huber_beta também está em escala log: 0,05 ≈ 5% de erro.
"""
import torch

from leaf_measures.data.normalization import MEASURE_ORDER, MEASURE_RANGES

LOSS_TYPES = ("mse", "huber", "mse_log", "huber_log", "l1_log")
_MIN_VALUE = 1e-3  # piso antes do log: a rede pode prever valores <= 0 no início do treino


def _huber(diff_abs, beta):
    return torch.where(diff_abs < beta, 0.5 * diff_abs ** 2 / beta, diff_abs - 0.5 * beta)


def _to_real_units(normalized):
    """(N, 4) normalizado 0–1 -> (N, 4) em cm / cm² (mesma conta de denormalize)."""
    lo = normalized.new_tensor([MEASURE_RANGES[m][0] for m in MEASURE_ORDER])
    hi = normalized.new_tensor([MEASURE_RANGES[m][1] for m in MEASURE_ORDER])
    return normalized * (hi - lo) + lo


def weighted_measures_loss(pred, target, weights, loss_type="mse", huber_beta=1.0):
    """
    pred, target: (N, 4) NORMALIZADOS, como saem da rede e de build_target_tensor.
    Retorna (loss_escalar, per_component_detached).
    """
    if loss_type in ("mse", "huber"):
        diff = pred - target
    elif loss_type in ("mse_log", "huber_log", "l1_log"):
        p = _to_real_units(pred).clamp(min=_MIN_VALUE)
        t = _to_real_units(target).clamp(min=_MIN_VALUE)
        diff = torch.log(p) - torch.log(t)
    else:
        raise ValueError(f"loss.type desconhecido: '{loss_type}'. Opções: {LOSS_TYPES}")

    if loss_type in ("mse", "mse_log"):
        per_sample = diff ** 2
    elif loss_type in ("huber", "huber_log"):
        per_sample = _huber(diff.abs(), huber_beta)
    else:  # l1_log
        per_sample = diff.abs()

    per_component = per_sample.mean(dim=0)
    return (per_component * weights).sum(), per_component.detach()
