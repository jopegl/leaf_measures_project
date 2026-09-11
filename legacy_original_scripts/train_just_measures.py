from datetime import datetime

# PARAMETROS
NUM_CLASSES = 3
BACKBONE = 'xception'
OUTPUT_STRIDE = 16
SYNC_BN = None
FREEZE_BN = True

LEARNING_RATE = 0.00002
MOMENTUM = 0.9
WEIGHT_DECAY = 4e-5

ITERS = 65000
POWER = 0.9
NUM_EPOCHS = 100

VAL_INTERVAL = 500
BATCH_SIZE = 2

NUM_WORKERS = 4
PIN_MEMORY = True

LENGTH_MIN = 4.0
LENGTH_MAX = 6.9
WIDTH_MIN = 4.8
WIDTH_MAX = 9.7
PERIMETER_MIN = 18.0
PERIMETER_MAX = 31.5

AREA_MIN = 14.4
AREA_MAX = 42.133

import sys
import os
import glob
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_SEG_DTYPE = np.uint8


def _strip_known_suffix(name: str) -> str:
    for suffix in ("_area", "_segmentation", "_annotation", "_image"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _parse_annotation(xml_path: str) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.findtext("image-name")
    obs = root.findtext("observation", default="")

    size_node = root.find("image-size")
    height = int(size_node.findtext("height"))
    width = int(size_node.findtext("width"))

    objects = root.find("objects")

    leaf_number = objects.findtext("leaf-number")

    marker = objects.find("marker")
    marker_real_area = float(marker.findtext("marker-real-area"))
    marker_projected_area = float(marker.findtext("marker-projected-area"))

    leaf = objects.find("leaf")
    dims = leaf.find("dimensions")

    leaf_area = float(dims.findtext("leaf-area"))
    leaf_width = float(dims.findtext("leaf-width"))
    leaf_length = float(dims.findtext("leaf-length"))
    leaf_perimeter = float(dims.findtext("leaf-perimeter"))

    return {
        "image_name": image_name,
        "observation": obs,
        "height": height,
        "width": width,
        "leaf_number": leaf_number,
        "marker_real_area": marker_real_area,
        "marker_projected_area": marker_projected_area,
        "leaf_area": leaf_area,
        "leaf_width": leaf_width,
        "leaf_length": leaf_length,
        "leaf_perimeter": leaf_perimeter,
    }


class LSIDBeansDataset(Dataset):
    """
    Versão sem o mapa de área por pixel (.raw). A área agora vem do
    mesmo XML de anotação de onde vêm length/width/perimeter
    (campo 'leaf-area' dentro de <dimensions>).
    """

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

        self.samples = self._index_samples()

        if split_file is not None:
            keep_ids = self._load_split_ids(split_file)
            self.samples = [
                s for s in self.samples
                if s["id"] in keep_ids
            ]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em '{root_dir}' "
                f"(split_file={split_file}). Confira os caminhos."
            )

    def _index_samples(self):
        samples = []

        xml_paths = sorted(
            glob.glob(
                os.path.join(
                    self.root_dir,
                    "*",
                    "annotation",
                    "*.xml"
                )
            )
        )

        for xml_path in xml_paths:

            leaf_folder = os.path.basename(
                os.path.dirname(
                    os.path.dirname(xml_path)
                )
            )

            basename = os.path.splitext(
                os.path.basename(xml_path)
            )[0]

            sample_id = _strip_known_suffix(basename)

            image_path = os.path.join(
                self.root_dir,
                leaf_folder,
                "images",
                sample_id + ".jpg"
            )

            seg_path = os.path.join(
                self.root_dir,
                leaf_folder,
                "segmentation",
                sample_id + "_segmentation.raw"
            )

            if not (
                os.path.isfile(image_path)
                and os.path.isfile(seg_path)
            ):
                continue

            samples.append(
                {
                    "id": sample_id,
                    "leaf_folder": leaf_folder,
                    "xml_path": xml_path,
                    "image_path": image_path,
                    "seg_path": seg_path,
                }
            )

        return samples

    @staticmethod
    def _load_split_ids(split_file: str) -> set:
        ids = set()

        if split_file.endswith(".csv"):
            df = pd.read_csv(
                split_file,
                header=None
            )

            for item in df[0]:
                ids.add(
                    _strip_known_suffix(str(item))
                )

        else:
            with open(split_file, "r") as f:
                for line in f:
                    line = line.strip()

                    if line:
                        line = os.path.splitext(line)[0]
                        ids.add(
                            _strip_known_suffix(line)
                        )

        return ids

    def _load_raw(self, path: str, dtype) -> np.ndarray:

        arr = np.fromfile(
            path,
            dtype=dtype
        )

        expected = self.height * self.width

        if arr.size != expected:
            raise ValueError(
                f"'{path}' tem {arr.size} elementos, esperado "
                f"{expected} ({self.height}x{self.width}). "
                f"Dtype errado? "
                f"Tamanho do arquivo = {os.path.getsize(path)} bytes."
            )

        return arr.reshape(
            self.height,
            self.width
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        ann = _parse_annotation(
            sample["xml_path"]
        )

        # Imagem
        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        image = TF.to_tensor(image)

        # Máscara de segmentação (classes por pixel)
        seg = self._load_raw(
            sample["seg_path"],
            self.seg_dtype
        ).astype(np.int64)

        seg_mask = torch.from_numpy(seg)

        if self.augment:
            image, seg_mask = self._joint_augment(
                image,
                seg_mask,
            )

        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,
            "seg_mask": seg_mask,

            # área agora vem do XML de anotação (mesmo lugar de length/width/perimeter)
            "leaf_area": torch.tensor(
                ann["leaf_area"],
                dtype=torch.float32
            ),

            "leaf_width": torch.tensor(
                ann["leaf_width"],
                dtype=torch.float32
            ),

            "leaf_length": torch.tensor(
                ann["leaf_length"],
                dtype=torch.float32
            ),

            "leaf_perimeter": torch.tensor(
                ann["leaf_perimeter"],
                dtype=torch.float32
            ),

            "leaf_number": sample["leaf_folder"],
            "image_name": ann["image_name"],
        }

    @staticmethod
    def _joint_augment(
        image,
        seg_mask,
    ):

        k = int(
            np.random.choice([0, 1, 2, 3])
        )

        if k > 0:
            image = torch.rot90(
                image,
                k,
                dims=[1, 2]
            )

            seg_mask = torch.rot90(
                seg_mask,
                k,
                dims=[0, 1]
            )

        if np.random.rand() < 0.5:

            image = torch.flip(
                image,
                dims=[2]
            )

            seg_mask = torch.flip(
                seg_mask,
                dims=[1]
            )

        if np.random.rand() < 0.5:

            image = torch.flip(
                image,
                dims=[1]
            )

            seg_mask = torch.flip(
                seg_mask,
                dims=[0]
            )

        return image, seg_mask




def main():

    ROOT = r"Leaf on Stem Image Dataset Beans (LSID-Beans)\LSID-Beans"

    TRAIN_CSV = (
        r"Leaf on Stem Image Dataset Beans (LSID-Beans)"
        r"\cross-validation\cv_1\split_1\train.csv"
    )

    TEST_CSV = (
        r"Leaf on Stem Image Dataset Beans (LSID-Beans)"
        r"\cross-validation\cv_1\split_1\test.csv"
    )

    # Dataset usado para TREINAMENTO.
    # Possui augmentation.
    train_dataset = LSIDBeansDataset(
        root_dir=ROOT,
        split_file=TRAIN_CSV,
        augment=True,
    )

    # Dataset de validação/teste.
    test_dataset = LSIDBeansDataset(
        root_dir=ROOT,
        split_file=TEST_CSV,
        augment=False,
    )

    print(f"Imagens de treino: {len(train_dataset)}")
    print(f"Imagens de validação: {len(test_dataset)}")

    from models.deeplab_just_measures import DeepLab

    model = DeepLab(
        num_classes=NUM_CLASSES,
        backbone=BACKBONE,
        output_stride=OUTPUT_STRIDE,
        sync_bn=SYNC_BN,
        freeze_bn=FREEZE_BN,
    )

    train_params = [
        {
            'params': model.get_1x_lr_params(),
            'lr': LEARNING_RATE
        },
        {
            'params': model.get_10x_lr_params(),
            'lr': LEARNING_RATE * 10
        }
    ]

    optimizer = torch.optim.SGD(
        train_params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=ITERS,
        power=POWER
    )

    # Loader usado no treinamento.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Loader de validação.
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    def calculate_miou(
        pred,
        target,
        num_classes
    ):

        pred = torch.argmax(
            pred,
            dim=1
        )

        ious = []

        for cls in range(num_classes):

            pred_cls = pred == cls
            target_cls = target == cls

            intersection = (
                pred_cls & target_cls
            ).sum().float()

            union = (
                pred_cls | target_cls
            ).sum().float()

            if union == 0:
                continue

            iou = intersection / union
            ious.append(iou)

        if len(ious) == 0:
            return 0.0

        return torch.stack(ious).mean().item()

    def calculate_measure_metrics(pred, target, eps=1e-8):

        error = pred - target

        # MAE
        mae = error.abs().mean().item()

        # RMSE
        rmse = torch.sqrt(
            (error ** 2).mean()
        ).item()

        # RER (%)
        rer = (
            error.abs()
            / (target.abs() + eps)
            * 100.0
        ).mean().item()

        # R²
        ss_res = (
            (target - pred) ** 2
        ).sum()

        ss_tot = (
            (target - target.mean()) ** 2
        ).sum()

        r2 = (
            1.0 - ss_res / (ss_tot + eps)
        ).item()

        return mae, rmse, rer, r2

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    MEASURE_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 5.0], device=device)

    def weighted_measures_loss(pred, target, weights):

        per_component_mse = ((pred - target) ** 2).mean(dim=0) 
        return (per_component_mse * weights).sum()

    model = model.to(device)

    criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_measures = torch.nn.MSELoss()

    best_val_loss = float("inf")
    iteration = 0
    checkpoint_path = "best_model_just_measures.pth"

    if os.path.exists(checkpoint_path):

        print(
            f"Checkpoint encontrado: "
            f"{checkpoint_path}"
        )

        print(
            "Carregando checkpoint..."
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        iteration = checkpoint[
            "iteration"
        ]

        best_val_loss = checkpoint[
            "val_loss"
        ]

        print(
            "Checkpoint carregado!"
        )

        print(
            f"Continuando da iteração: "
            f"{iteration}"
        )

        print(
            f"Melhor validation loss: "
            f"{best_val_loss:.4f}"
        )

    else:

        print(
            "Nenhum checkpoint encontrado."
        )

        print(
            "Iniciando treinamento do zero."
        )

    print(
        f"Começando "
        f"[{datetime.now().strftime('%H:%M:%S')}]"
    )


    while iteration < ITERS:

        model.train()

        for batch in train_loader:

            if iteration >= ITERS:
                break

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            seg_mask = batch["seg_mask"].to(
                device,
                non_blocking=True
            )

            length = batch[
                "leaf_length"
            ].to(
                device,
                non_blocking=True
            )

            width = batch[
                "leaf_width"
            ].to(
                device,
                non_blocking=True
            )

            perimeter = batch[
                "leaf_perimeter"
            ].to(
                device,
                non_blocking=True
            )

            area = batch[
                "leaf_area"
            ].to(
                device,
                non_blocking=True
            )

            length = (
                length - LENGTH_MIN
            ) / (
                LENGTH_MAX - LENGTH_MIN
            )

            width = (
                width - WIDTH_MIN
            ) / (
                WIDTH_MAX - WIDTH_MIN
            )

            perimeter = (
                perimeter - PERIMETER_MIN
            ) / (
                PERIMETER_MAX - PERIMETER_MIN
            )

            area = (
                area - AREA_MIN
            ) / (
                AREA_MAX - AREA_MIN
            )

            measures_target = torch.stack(
                [
                    length,
                    width,
                    perimeter,
                    area
                ],
                dim=1
            )

            optimizer.zero_grad()

            seg_out, measures_out = model(images)

            if (
                torch.isnan(measures_out).any()
                or torch.isinf(measures_out).any()
            ):

                print(
                    f"[iter {iteration}] "
                    f"measures_out tem NaN/Inf! "
                    f"min={measures_out.min().item()}, "
                    f"max={measures_out.max().item()}"
                )

            loss_seg = criterion_seg(
                seg_out,
                seg_mask
            )

            loss_measures = weighted_measures_loss(measures_out, measures_target, MEASURE_WEIGHTS)

            loss = (
                loss_seg
                +
                loss_measures * 10
            )

            loss.backward()

            optimizer.step()
            scheduler.step()

            iteration += 1

            if iteration % 100 == 0:

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Iteration [{iteration}/{ITERS}] "
                    f"LR: {current_lr:.8f} "
                    f"Loss: {loss.item():.4f} "
                    f"Seg: {loss_seg.item():.4f} "
                    f"Measures: {loss_measures.item():.4f}"
                )

            if iteration % VAL_INTERVAL == 0:

                model.eval()

                val_length_preds = []
                val_length_targets = []

                val_width_preds = []
                val_width_targets = []

                val_perimeter_preds = []
                val_perimeter_targets = []

                val_area_preds = []
                val_area_targets = []

                val_total_loss = 0.0
                val_seg_loss = 0.0
                val_measures_loss = 0.0
                val_miou = 0.0

                with torch.no_grad():

                    for val_batch in test_loader:

                        val_images = val_batch[
                            "image"
                        ].to(
                            device,
                            non_blocking=True
                        )

                        val_seg_mask = val_batch[
                            "seg_mask"
                        ].to(
                            device,
                            non_blocking=True
                        )

                        val_length = (
                            val_batch[
                                "leaf_length"
                            ].to(device)
                            - LENGTH_MIN
                        ) / (
                            LENGTH_MAX - LENGTH_MIN
                        )

                        val_width = (
                            val_batch[
                                "leaf_width"
                            ].to(device)
                            - WIDTH_MIN
                        ) / (
                            WIDTH_MAX - WIDTH_MIN
                        )

                        val_perimeter = (
                            val_batch[
                                "leaf_perimeter"
                            ].to(device)
                            - PERIMETER_MIN
                        ) / (
                            PERIMETER_MAX - PERIMETER_MIN
                        )

                        val_area = (
                            val_batch[
                                "leaf_area"
                            ].to(device)
                            - AREA_MIN
                        ) / (
                            AREA_MAX - AREA_MIN
                        )

                        val_measures_target = torch.stack(
                            [
                                val_length,
                                val_width,
                                val_perimeter,
                                val_area
                            ],
                            dim=1
                        )

                        val_seg_out, val_measures_out = model(
                            val_images
                        )

                        val_loss_seg = criterion_seg(
                            val_seg_out,
                            val_seg_mask
                        )

                        val_loss_measures = criterion_measures(
                            val_measures_out,
                            val_measures_target
                        )

                        val_length_preds.extend(
                            val_measures_out[:, 0].detach().cpu().tolist()
                        )

                        val_length_targets.extend(
                            val_measures_target[:, 0].detach().cpu().tolist()
                        )

                        val_width_preds.extend(
                            val_measures_out[:, 1].detach().cpu().tolist()
                        )

                        val_width_targets.extend(
                            val_measures_target[:, 1].detach().cpu().tolist()
                        )

                        val_perimeter_preds.extend(
                            val_measures_out[:, 2].detach().cpu().tolist()
                        )

                        val_perimeter_targets.extend(
                            val_measures_target[:, 2].detach().cpu().tolist()
                        )

                        val_area_preds.extend(
                            val_measures_out[:, 3].detach().cpu().tolist()
                        )

                        val_area_targets.extend(
                            val_measures_target[:, 3].detach().cpu().tolist()
                        )

                        val_loss = (
                            val_loss_seg
                            +
                            val_loss_measures
                        )

                        val_total_loss += (
                            val_loss.item()
                        )

                        val_seg_loss += (
                            val_loss_seg.item()
                        )

                        val_measures_loss += (
                            val_loss_measures.item()
                        )

                        val_miou += calculate_miou(
                            val_seg_out,
                            val_seg_mask,
                            NUM_CLASSES
                        )

                val_total_loss /= len(test_loader)
                val_seg_loss /= len(test_loader)
                val_measures_loss /= len(test_loader)
                val_miou /= len(test_loader)

                val_length_preds = torch.tensor(val_length_preds)
                val_length_targets = torch.tensor(val_length_targets)

                val_width_preds = torch.tensor(val_width_preds)
                val_width_targets = torch.tensor(val_width_targets)

                val_perimeter_preds = torch.tensor(val_perimeter_preds)
                val_perimeter_targets = torch.tensor(val_perimeter_targets)

                val_area_preds = torch.tensor(val_area_preds)
                val_area_targets = torch.tensor(val_area_targets)

                val_length_preds = (
                    val_length_preds
                    * (LENGTH_MAX - LENGTH_MIN)
                    + LENGTH_MIN
                )

                val_length_targets = (
                    val_length_targets
                    * (LENGTH_MAX - LENGTH_MIN)
                    + LENGTH_MIN
                )

                val_width_preds = (
                    val_width_preds
                    * (WIDTH_MAX - WIDTH_MIN)
                    + WIDTH_MIN
                )

                val_width_targets = (
                    val_width_targets
                    * (WIDTH_MAX - WIDTH_MIN)
                    + WIDTH_MIN
                )

                val_perimeter_preds = (
                    val_perimeter_preds
                    * (PERIMETER_MAX - PERIMETER_MIN)
                    + PERIMETER_MIN
                )

                val_perimeter_targets = (
                    val_perimeter_targets
                    * (PERIMETER_MAX - PERIMETER_MIN)
                    + PERIMETER_MIN
                )

                val_area_preds = (
                    val_area_preds
                    * (AREA_MAX - AREA_MIN)
                    + AREA_MIN
                )

                val_area_targets = (
                    val_area_targets
                    * (AREA_MAX - AREA_MIN)
                    + AREA_MIN
                )

                length_mae, length_rmse, length_rer, length_r2 = \
                    calculate_measure_metrics(
                        val_length_preds,
                        val_length_targets
                    )

                width_mae, width_rmse, width_rer, width_r2 = \
                    calculate_measure_metrics(
                        val_width_preds,
                        val_width_targets
                    )

                perimeter_mae, perimeter_rmse, perimeter_rer, perimeter_r2 = \
                    calculate_measure_metrics(
                        val_perimeter_preds,
                        val_perimeter_targets
                    )

                area_mae, area_rmse, area_rer, area_r2 = \
                    calculate_measure_metrics(
                        val_area_preds,
                        val_area_targets
                    )

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                print("\n" + "=" * 60)

                print(f"Iteration [{iteration}/{ITERS}]")

                print(f"LR: {current_lr:.8f}")

                print(f"Validation Loss: {val_total_loss:.4f}")

                print(f"  Segmentation: {val_seg_loss:.4f}")

                print(f"  Measures:     {val_measures_loss:.4f}")

                print(f"  mIoU:         {val_miou:.4f}")

                print()

                print("  Val Measures:")

                print(
                    f"    Length:    "
                    f"MAE={length_mae:.4f}  "
                    f"RMSE={length_rmse:.4f}  "
                    f"RER={length_rer:.2f}%  "
                    f"R²={length_r2:.4f}"
                )

                print(
                    f"    Width:     "
                    f"MAE={width_mae:.4f}  "
                    f"RMSE={width_rmse:.4f}  "
                    f"RER={width_rer:.2f}%  "
                    f"R²={width_r2:.4f}"
                )

                print(
                    f"    Perimeter: "
                    f"MAE={perimeter_mae:.4f}  "
                    f"RMSE={perimeter_rmse:.4f}  "
                    f"RER={perimeter_rer:.2f}%  "
                    f"R²={perimeter_r2:.4f}"
                )

                print(
                    f"    Area:      "
                    f"MAE={area_mae:.4f}  "
                    f"RMSE={area_rmse:.4f}  "
                    f"RER={area_rer:.2f}%  "
                    f"R²={area_r2:.4f}"
                )

                print("=" * 60)

                if val_total_loss < best_val_loss:

                    best_val_loss = val_total_loss

                    torch.save(
                        {
                            "iteration": iteration,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "val_loss": val_total_loss,
                            "val_seg_loss": val_seg_loss,
                            "val_measures_loss": val_measures_loss,
                            "val_miou": val_miou,
                            "length_mae": length_mae,
                            "width_mae": width_mae,
                            "perimeter_mae": perimeter_mae,
                            "area_mae": area_mae,
                        },
                        checkpoint_path
                    )

                    print(">>> Melhor modelo salvo!")

                print()

    print("\nTreinamento finalizado.")

    print(f"Melhor validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()