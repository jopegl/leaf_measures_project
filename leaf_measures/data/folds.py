"""
Folds da cross-validation (cv_X/split_Y/{train,val,test}.csv).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

_FOLD_ID_RE = re.compile(r"^cv(\d+)_split(\d+)$")


@dataclass(frozen=True)
class Fold:
    cv: int
    split: int
    cv_root: str

    @property
    def id(self) -> str:
        return f"cv{self.cv}_split{self.split}"

    @property
    def dir(self) -> str:
        return os.path.join(self.cv_root, f"cv_{self.cv}", f"split_{self.split}")

    def csv(self, subset: str) -> str:
        """subset: 'train' | 'val' | 'test'"""
        return os.path.join(self.dir, f"{subset}.csv")


def parse_fold_id(fold_id: str) -> tuple[int, int]:
    m = _FOLD_ID_RE.match(fold_id.strip())
    if not m:
        raise ValueError(f"Fold inválido '{fold_id}'. Formato esperado: cv1_split3")
    return int(m.group(1)), int(m.group(2))


def folds_from_config(cfg, only: list[str] | None = None) -> list[Fold]:
    """Folds definidos em cfg.folds (cvs x splits), ou só os passados em `only`."""
    cv_root = cfg.data.cv_root
    if only:
        return [Fold(*parse_fold_id(f), cv_root=cv_root) for f in only]
    return [Fold(int(cv), int(s), cv_root) for cv in cfg.folds.cvs for s in cfg.folds.splits]


def discover_folds(cv_root: str) -> list[Fold]:
    """Todos os cv_*/split_* existentes no disco."""
    folds = []
    for cv_dir in glob.glob(os.path.join(cv_root, "cv_*")):
        m_cv = re.search(r"cv_(\d+)$", cv_dir)
        if not m_cv:
            continue
        for split_dir in glob.glob(os.path.join(cv_dir, "split_*")):
            m_s = re.search(r"split_(\d+)$", split_dir)
            if m_s:
                folds.append(Fold(int(m_cv.group(1)), int(m_s.group(1)), cv_root))
    return sorted(folds, key=lambda f: (f.cv, f.split))
