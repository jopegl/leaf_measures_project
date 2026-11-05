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


METRIC_KEYS = ("mae", "rmse", "rer", "r2")


def compute_metrics(preds_by_measure: dict, targets_by_measure: dict) -> dict:
    """{medida: tensor} x2 (desnormalizados) -> {medida: {mae, rmse, rer, r2}}."""
    out = {}
    for name, preds in preds_by_measure.items():
        mae, rmse, rer, r2 = calculate_measure_metrics(preds, targets_by_measure[name])
        out[name] = {"mae": mae, "rmse": rmse, "rer": rer, "r2": r2}
    return out


def weighted_average(entries: list) -> dict:
    """
    Média ponderada pelo nº de amostras de teste.
    entries: [{'n_samples': int, 'metrics': {medida: {mae,...}}}, ...]
    """
    total = sum(e["n_samples"] for e in entries)
    if total == 0:
        return {}
    measures = entries[0]["metrics"].keys()
    return {
        name: {
            k: sum(e["metrics"][name][k] * e["n_samples"] for e in entries) / total
            for k in METRIC_KEYS
        }
        for name in measures
    }
