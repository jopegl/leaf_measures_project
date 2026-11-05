import torch


def weighted_measures_loss(pred, target, weights, loss_type="mse", huber_beta=1.0):
    """
    Loss ponderada por componente (length, width, perimeter, area).
    loss_type: 'mse' ou 'huber' (SmoothL1).
    Retorna (loss_escalar, per_component_detached).
    """
    if loss_type == "mse":
        per_component = ((pred - target) ** 2).mean(dim=0)
    elif loss_type == "huber":
        diff = (pred - target).abs()
        per_sample = torch.where(
            diff < huber_beta,
            0.5 * diff ** 2 / huber_beta,
            diff - 0.5 * huber_beta,
        )
        per_component = per_sample.mean(dim=0)
    else:
        raise ValueError(f"loss_type desconhecido: {loss_type}")

    return (per_component * weights).sum(), per_component.detach()
