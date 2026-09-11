import torch


def calculate_measure_metrics(pred, target, eps=1e-8):
    """MAE, RMSE, RER(%) e R² para uma medida de regressão (já desnormalizada)."""
    error = pred - target

    mae = error.abs().mean().item()
    rmse = torch.sqrt((error ** 2).mean()).item()
    rer = (error.abs() / (target.abs() + eps) * 100.0).mean().item()

    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / (ss_tot + eps)).item()

    return mae, rmse, rer, r2


def calculate_miou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return torch.stack(ious).mean().item()
