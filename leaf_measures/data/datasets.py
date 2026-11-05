"""
Dataset do LSID-Beans para regressão das medidas da folha.

Um único dataset, `LSIDBeansDataset`, cobre todos os casos:
- load_masks=False -> só imagem + medidas (não lê o .raw de segmentação).
- load_masks=True  -> também devolve `masks` [2, H, W] (canal 0 = folha,
                      canal 1 = marcador), usado pelos modos de entrada que
                      precisam de máscara (hadamard, mask_concat...).

O dataset devolve a imagem em [0, 1], sem normalização. A normalização
ImageNet (opcional) e a montagem do tensor de entrada são feitas em
leaf_measures/models/inputs.py, para que todos os modos usem o mesmo caminho.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from leaf_measures.data.annotation import parse_annotation, strip_known_suffix
from leaf_measures.data.splits import load_split_ids

LEAF_ID = 1
MARKER_ID = 2
SEG_DTYPE = np.uint8


def index_samples(root_dir: str, require_seg: bool = False) -> list[dict]:
    """Procura <root_dir>/*/annotation/*.xml e casa com a imagem (e o .raw, se pedido)."""
    samples = []
    xml_paths = sorted(glob.glob(os.path.join(root_dir, "*", "annotation", "*.xml")))

    for xml_path in xml_paths:
        leaf_folder = os.path.basename(os.path.dirname(os.path.dirname(xml_path)))
        sample_id = strip_known_suffix(os.path.splitext(os.path.basename(xml_path))[0])

        image_path = os.path.join(root_dir, leaf_folder, "images", sample_id + ".jpg")
        if not os.path.isfile(image_path):
            continue

        sample = {
            "id": sample_id,
            "leaf_folder": leaf_folder,
            "xml_path": xml_path,
            "image_path": image_path,
        }
        if require_seg:
            seg_path = os.path.join(root_dir, leaf_folder, "segmentation", sample_id + "_segmentation.raw")
            if not os.path.isfile(seg_path):
                continue
            sample["seg_path"] = seg_path
        samples.append(sample)
    return samples


class LSIDBeansDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split_file: str | None = None,
        load_masks: bool = False,
        augment: bool = False,
        height: int = 512,
        width: int = 512,
    ):
        self.root_dir = root_dir
        self.load_masks = load_masks
        self.augment = augment
        self.height = height
        self.width = width

        self.samples = index_samples(root_dir, require_seg=load_masks)
        if split_file is not None:
            keep = load_split_ids(split_file)
            self.samples = [s for s in self.samples if s["id"] in keep]

        if not self.samples:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em '{root_dir}' (split_file={split_file}, "
                f"load_masks={load_masks}). Confira os caminhos em data.root / data.cv_root."
            )

    def __len__(self):
        return len(self.samples)

    def _load_seg(self, path: str) -> torch.Tensor:
        arr = np.fromfile(path, dtype=SEG_DTYPE)
        expected = self.height * self.width
        if arr.size != expected:
            raise ValueError(
                f"'{path}' tem {arr.size} elementos, esperado {expected} "
                f"({self.height}x{self.width}). Tamanho do arquivo = {os.path.getsize(path)} bytes."
            )
        seg = torch.from_numpy(arr.reshape(self.height, self.width).astype(np.int64))
        return torch.stack([seg == LEAF_ID, seg == MARKER_ID], dim=0).float()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = parse_annotation(sample["xml_path"])

        image = TF.to_tensor(Image.open(sample["image_path"]).convert("RGB"))
        masks = self._load_seg(sample["seg_path"]) if self.load_masks else None

        if self.augment:
            image, masks = _random_dihedral(image, masks)

        item = {
            "image": image,
            "leaf_area": torch.tensor(ann["leaf_area"], dtype=torch.float32),
            "leaf_width": torch.tensor(ann["leaf_width"], dtype=torch.float32),
            "leaf_length": torch.tensor(ann["leaf_length"], dtype=torch.float32),
            "leaf_perimeter": torch.tensor(ann["leaf_perimeter"], dtype=torch.float32),
            "leaf_number": sample["leaf_folder"],
            "image_name": ann["image_name"],
        }
        if masks is not None:
            item["masks"] = masks
        return item


def _random_dihedral(image: torch.Tensor, masks: torch.Tensor | None):
    """Rotação de 90°·k + flips horizontal/vertical, aplicados igualmente à imagem e às máscaras."""
    k = int(np.random.choice([0, 1, 2, 3]))
    flip_h = np.random.rand() < 0.5
    flip_v = np.random.rand() < 0.5

    def tf(t):
        if t is None:
            return None
        if k > 0:
            t = torch.rot90(t, k, dims=[1, 2])
        if flip_h:
            t = torch.flip(t, dims=[2])
        if flip_v:
            t = torch.flip(t, dims=[1])
        return t

    return tf(image), tf(masks)
