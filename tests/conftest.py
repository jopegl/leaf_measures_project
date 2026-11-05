"""Gera um mini dataset falso no formato do LSID-Beans para testar o pipeline."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

XML = """<annotation>
  <image-name>{name}.jpg</image-name>
  <observation></observation>
  <image-size><height>{h}</height><width>{w}</width></image-size>
  <objects>
    <leaf-number>{leaf}</leaf-number>
    <marker><marker-real-area>4.0</marker-real-area><marker-projected-area>100.0</marker-projected-area></marker>
    <leaf><dimensions>
      <leaf-area>{area}</leaf-area><leaf-width>{width}</leaf-width>
      <leaf-length>{length}</leaf-length><leaf-perimeter>{perim}</leaf-perimeter>
    </dimensions></leaf>
  </objects>
</annotation>
"""

SIZE = 64


@pytest.fixture
def fake_dataset(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    root = tmp_path / "LSID-Beans"
    cv_root = tmp_path / "cross-validation"
    ids = []
    for leaf in ("leaf_01", "leaf_02"):
        for d in ("annotation", "images", "segmentation"):
            (root / leaf / d).mkdir(parents=True)
        for i in range(12):
            sid = f"{leaf}_img{i}"
            ids.append(sid)
            (root / leaf / "annotation" / f"{sid}_annotation.xml").write_text(XML.format(
                name=sid, h=SIZE, w=SIZE, leaf=leaf, area=rng.uniform(15, 40), width=rng.uniform(5, 9),
                length=rng.uniform(4, 6.5), perim=rng.uniform(19, 30)))
            Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)).save(
                root / leaf / "images" / f"{sid}.jpg")
            seg = np.zeros((SIZE, SIZE), np.uint8)
            seg[10:40, 10:40] = 1
            seg[45:55, 45:55] = 2
            seg.tofile(root / leaf / "segmentation" / f"{sid}_segmentation.raw")

    split_dir = cv_root / "cv_1" / "split_1"
    split_dir.mkdir(parents=True)
    for name, part in (("train", ids[:8]), ("val", ids[8:10]), ("test", ids[10:12])):
        (split_dir / f"{name}.csv").write_text("\n".join(f"{s}.jpg" for s in part) + "\n")
    return {"root": str(root), "cv_root": str(cv_root), "tmp": tmp_path, "ids": ids}


def make_overrides(fake, **extra):
    base = {
        "data.root": fake["root"],
        "data.cv_root": fake["cv_root"],
        "data.height": SIZE,
        "data.width": SIZE,
        "experiment.output_dir": str(fake["tmp"] / "runs"),
        "eval.results_dir": str(fake["tmp"] / "results"),
        "folds.cvs": "[1]",
        "folds.splits": "[1]",
        "model.backbone": "resnet18",
        "model.pretrained": "false",
        "train.iters": 4,
        "train.val_interval": 2,
        "train.log_every": 2,
        "train.freeze_backbone_validations": 1,
        "train.num_workers": 0,
        "train.batch_size": 2,
        "eval.batch_size": 2,
    }
    base.update(extra)
    return [f"{k}={v}" for k, v in base.items()]
