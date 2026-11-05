"""
Carregamento de configurações de experimento (YAML).

- Cada experimento é um YAML em configs/experiments/.
- Um YAML pode herdar de outro com a chave `_base_` (caminho relativo ao próprio
  arquivo). Os dicionários são mesclados recursivamente; o arquivo filho vence.
- Qualquer campo pode ser sobrescrito pela linha de comando com
  `--set chave.subchave=valor` (ex.: `--set model.backbone=resnet18 train.iters=1000`).
- Se `experiment.name` não for definido, usa o nome do arquivo YAML.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml


class Config(dict):
    """dict com acesso por atributo (cfg.train.lr_head), recursivo."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"Config não tem a chave '{key}'") from exc

    def __setattr__(self, key, value):
        self[key] = Config(value) if isinstance(value, dict) else value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(self.to_dict(), memo))

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_base(path: str, _seen: tuple = ()) -> dict:
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"Herança circular de config: {' -> '.join(_seen + (path,))}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base_ref = data.pop("_base_", None)
    if base_ref is None:
        return data
    bases = base_ref if isinstance(base_ref, list) else [base_ref]
    merged: dict = {}
    for b in bases:
        b_path = os.path.join(os.path.dirname(path), b)
        merged = _deep_merge(merged, _load_yaml_with_base(b_path, _seen + (path,)))
    return _deep_merge(merged, data)


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    # PyYAML lê "1e-4" como string; aqui convertemos números em notação científica.
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(data: dict, overrides: Iterable[str]) -> dict:
    data = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override inválido '{item}'. Use chave.subchave=valor")
        key, raw = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Override '{item}': '{p}' não é uma seção")
        node[parts[-1]] = _parse_value(raw)
    return data


def _coerce_numbers(node):
    """Converte strings numéricas (ex.: '1e-4' vindo do YAML) em float."""
    if isinstance(node, dict):
        return {k: _coerce_numbers(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce_numbers(v) for v in node]
    if isinstance(node, str):
        try:
            return float(node) if any(c in node.lower() for c in ".e") and node[0] in "+-.0123456789" else node
        except ValueError:
            return node
    return node


def load_config(path: str, overrides: Iterable[str] = ()) -> Config:
    data = _load_yaml_with_base(path)
    data = apply_overrides(data, overrides)
    data = _coerce_numbers(data)

    data.setdefault("experiment", {})
    if not data["experiment"].get("name"):
        data["experiment"]["name"] = os.path.splitext(os.path.basename(path))[0]

    cfg = Config(data)
    _validate(cfg)
    return cfg


REQUIRED_SECTIONS = ("experiment", "data", "folds", "input", "model", "train", "loss", "eval")


def _validate(cfg: Config):
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(
            f"Config incompleta, faltam as seções {missing}. "
            "O YAML do experimento deveria herdar de configs/base.yaml (_base_: ../base.yaml)."
        )


def save_config(cfg: Config, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
