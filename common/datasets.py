"""
Datasets do LSID-Beans.

- LSIDBeansMeasuresDataset: só regressão (length/width/perimeter/area), sem
  carregar o .raw de segmentação. Usado pelo ResNetRegressor.
- LSIDBeansDataset: regressão + segmentação (carrega o .raw). Usado pelo
  DeepLab multi-tarefa.

Ambos aceitam um split_file (train.csv / val.csv / test.csv) para filtrar
quais amostras entram no dataset.
"""
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from common.annotation import parse_annotation, strip_known_suffix
from common.splits import load_split_ids

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_SEG_DTYPE = np.uint8


def _index_samples(root_dir: str):
    """Indexação comum: procura todo <root_dir>/*/annotation/*.xml e casa com a imagem."""
    samples = []
    xml_paths = sorted(glob.glob(os.path.join(root_dir, "*", "annotation", "*.xml")))

    for xml_path in xml_paths:
        leaf_folder = os.path.basename(os.path.dirname(os.path.dirname(xml_path)))
        basename = os.path.splitext(os.path.basename(xml_path))[0]
        sample_id = strip_known_suffix(basename)

        image_path = os.path.join(root_dir, leaf_folder, "images", sample_id + ".jpg")
        if not os.path.isfile(image_path):
            continue

        samples.append(
            {
                "id": sample_id,
                "leaf_folder": leaf_folder,
                "xml_path": xml_path,
                "image_path": image_path,
            }
        )
    return samples


class LSIDBeansMeasuresDataset(Dataset):
    """Versão só-regressão: não carrega .raw de segmentação."""

    def __init__(
        self,
        root_dir: str,
        split_file: str = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        augment: bool = False,
        image_transform=None,
    ):
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.augment = augment
        self.image_transform = image_transform

        self.samples = _index_samples(root_dir)

        if split_file is not None:
            keep_ids = load_split_ids(split_file)
            self.samples = [s for s in self.samples if s["id"] in keep_ids]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em '{root_dir}' "
                f"(split_file={split_file}). Confira os caminhos."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = parse_annotation(sample["xml_path"])

        image = Image.open(sample["image_path"]).convert("RGB")
        image = TF.to_tensor(image)

        if self.augment:
            image = self._augment_image(image)

        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,
            "leaf_area": torch.tensor(ann["leaf_area"], dtype=torch.float32),
            "leaf_width": torch.tensor(ann["leaf_width"], dtype=torch.float32),
            "leaf_length": torch.tensor(ann["leaf_length"], dtype=torch.float32),
            "leaf_perimeter": torch.tensor(ann["leaf_perimeter"], dtype=torch.float32),
            "leaf_number": sample["leaf_folder"],
            "image_name": ann["image_name"],
        }

    @staticmethod
    def _augment_image(image):
        k = int(np.random.choice([0, 1, 2, 3]))
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        if np.random.rand() < 0.5:
            image = torch.flip(image, dims=[2])
        if np.random.rand() < 0.5:
            image = torch.flip(image, dims=[1])
        return image


class LSIDBeansDataset(Dataset):
    """Versão com segmentação (carrega o .raw) + regressão. Usada pelo DeepLab."""

    def __init__(
        self,
        root_dir: str,
        split_file: str = None,
        seg_dtype=DEFAULT_SEG_DTYPE,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        augment: bool = False,
        image_transform=None,
    ):
        self.root_dir = root_dir
        self.seg_dtype = seg_dtype
        self.height = height
        self.width = width
        self.augment = augment
        self.image_transform = image_transform

        self.samples = self._index_samples_with_seg(root_dir)

        if split_file is not None:
            keep_ids = load_split_ids(split_file)
            self.samples = [s for s in self.samples if s["id"] in keep_ids]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em '{root_dir}' "
                f"(split_file={split_file}). Confira os caminhos."
            )

    def _index_samples_with_seg(self, root_dir):
        base_samples = _index_samples(root_dir)
        samples = []
        for s in base_samples:
            seg_path = os.path.join(
                root_dir, s["leaf_folder"], "segmentation", s["id"] + "_segmentation.raw"
            )
            if not os.path.isfile(seg_path):
                continue
            s = dict(s)
            s["seg_path"] = seg_path
            samples.append(s)
        return samples

    def _load_raw(self, path: str, dtype) -> np.ndarray:
        arr = np.fromfile(path, dtype=dtype)
        expected = self.height * self.width
        if arr.size != expected:
            raise ValueError(
                f"'{path}' tem {arr.size} elementos, esperado "
                f"{expected} ({self.height}x{self.width}). Dtype errado? "
                f"Tamanho do arquivo = {os.path.getsize(path)} bytes."
            )
        return arr.reshape(self.height, self.width)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = parse_annotation(sample["xml_path"])

        image = Image.open(sample["image_path"]).convert("RGB")
        image = TF.to_tensor(image)

        seg = self._load_raw(sample["seg_path"], self.seg_dtype).astype(np.int64)
        seg_mask = torch.from_numpy(seg)

        if self.augment:
            image, seg_mask = self._joint_augment(image, seg_mask)

        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,
            "seg_mask": seg_mask,
            "leaf_area": torch.tensor(ann["leaf_area"], dtype=torch.float32),
            "leaf_width": torch.tensor(ann["leaf_width"], dtype=torch.float32),
            "leaf_length": torch.tensor(ann["leaf_length"], dtype=torch.float32),
            "leaf_perimeter": torch.tensor(ann["leaf_perimeter"], dtype=torch.float32),
            "leaf_number": sample["leaf_folder"],
            "image_name": ann["image_name"],
        }

    @staticmethod
    def _joint_augment(image, seg_mask):
        k = int(np.random.choice([0, 1, 2, 3]))
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            seg_mask = torch.rot90(seg_mask, k, dims=[0, 1])

        if np.random.rand() < 0.5:
            image = torch.flip(image, dims=[2])
            seg_mask = torch.flip(seg_mask, dims=[1])

        if np.random.rand() < 0.5:
            image = torch.flip(image, dims=[1])
            seg_mask = torch.flip(seg_mask, dims=[0])

        return image, seg_mask

class LSIDBeansMaskedMeasuresDataset(LSIDBeansDataset):
    LEAF_ID = 1
    MARKER_ID = 2

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        seg = item.pop("seg_mask")
        item["masks"] = torch.stack(
            [(seg == self.LEAF_ID), (seg == self.MARKER_ID)], dim=0
        ).float()
        return item
