"""
Normalização min-max das medidas de folha.
Centralizado aqui para não duplicar os mesmos MIN/MAX em cada script de treino.
"""
import torch

MEASURE_RANGES = {
    "length": (4.0, 6.9),
    "width": (4.8, 9.7),
    "perimeter": (18.0, 31.5),
    "area": (14.4, 42.133),
}

# Ordem fixa usada em todos os tensores empilhados (measures_target, measures_out, ...)
MEASURE_ORDER = ["length", "width", "perimeter", "area"]


def normalize(name: str, value: torch.Tensor) -> torch.Tensor:
    lo, hi = MEASURE_RANGES[name]
    return (value - lo) / (hi - lo)


def denormalize(name: str, value: torch.Tensor) -> torch.Tensor:
    lo, hi = MEASURE_RANGES[name]
    return value * (hi - lo) + lo


def build_target_tensor(batch: dict, device) -> torch.Tensor:
    """Monta o tensor (N, 4) de targets normalizados, na ordem MEASURE_ORDER."""
    cols = []
    for name in MEASURE_ORDER:
        key = f"leaf_{name}"
        v = batch[key].to(device, non_blocking=True)
        cols.append(normalize(name, v))
    return torch.stack(cols, dim=1)


def denormalize_columns(tensor_n4: torch.Tensor) -> dict:
    """Recebe um tensor (N, 4) NORMALIZADO (cpu) e devolve dict {nome: tensor desnormalizado}."""
    out = {}
    for i, name in enumerate(MEASURE_ORDER):
        out[name] = denormalize(name, tensor_n4[:, i])
    return out
