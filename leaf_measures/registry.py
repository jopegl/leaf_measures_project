"""
Registro simples por nome. Permite adicionar um novo modelo ou um novo modo de
entrada só com um decorator, sem mexer em train.py / evaluate.py:

    from leaf_measures.registry import MODELS

    @MODELS.register("meu_modelo")
    class MeuModelo(nn.Module): ...

e no YAML:  model: {name: meu_modelo, ...}
"""
from __future__ import annotations


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict = {}

    def register(self, name: str, **meta):
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' já registrado")
            self._items[name] = (obj, meta)
            return obj

        return deco

    def get(self, name: str):
        return self._entry(name)[0]

    def meta(self, name: str) -> dict:
        return self._entry(name)[1]

    def names(self) -> list[str]:
        return sorted(self._items)

    def _entry(self, name):
        if name not in self._items:
            raise KeyError(f"{self.kind} '{name}' desconhecido. Disponíveis: {self.names()}")
        return self._items[name]


MODELS = Registry("modelo")
INPUT_MODES = Registry("modo de entrada")
