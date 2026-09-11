from datetime import datetime
import sys
import os
import glob
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# =========================================================
# PARAMETROS
# =========================================================
# Backbone da ResNet: 'resnet18', 'resnet34', 'resnet50' ...
BACKBONE = 'resnet50'
PRETRAINED = True
FREEZE_BACKBONE_EPOCHS = 2   # nº de "epochs" (em termos de VAL_INTERVAL) com backbone congelado antes de liberar fine-tuning

NUM_OUTPUTS = 4  # length, width, perimeter, area

# LRs diferenciadas: backbone pré-treinado usa LR menor, head (novo) usa LR maior
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

ITERS = 20000          # regressão converge bem mais rápido que segmentação multi-tarefa
VAL_INTERVAL = 250
BATCH_SIZE = 16         # sem o mapa de segmentação em memória, dá pra subir bem o batch

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

# Pesos da loss por medida (length, width, perimeter, area)
MEASURE_WEIGHTS_VALUES = [1.0, 1.0, 1.0, 5.0]

# Tipo de loss de regressão: 'mse' ou 'huber' (SmoothL1)
REGRESSION_LOSS = 'huber'
HUBER_BETA = 0.1  # só usado se REGRESSION_LOSS == 'huber'


DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512


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


class LSIDBeansMeasuresDataset(Dataset):
    """
    Versão só-regressão: não carrega mais o .raw de segmentação (não é
    necessário e evita I/O e memória desperdiçados agora que não há
    cabeça de segmentação).
    """

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

        self.samples = self._index_samples()

        if split_file is not None:
            keep_ids = self._load_split_ids(split_file)
            self.samples = [s for s in self.samples if s["id"] in keep_ids]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em '{root_dir}' "
                f"(split_file={split_file}). Confira os caminhos."
            )

    def _index_samples(self):
        samples = []

        xml_paths = sorted(
            glob.glob(os.path.join(self.root_dir, "*", "annotation", "*.xml"))
        )

        for xml_path in xml_paths:
            leaf_folder = os.path.basename(os.path.dirname(os.path.dirname(xml_path)))
            basename = os.path.splitext(os.path.basename(xml_path))[0]
            sample_id = _strip_known_suffix(basename)

            image_path = os.path.join(self.root_dir, leaf_folder, "images", sample_id + ".jpg")

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

    @staticmethod
    def _load_split_ids(split_file: str) -> set:
        ids = set()

        if split_file.endswith(".csv"):
            df = pd.read_csv(split_file, header=None)
            for item in df[0]:
                ids.add(_strip_known_suffix(str(item)))
        else:
            with open(split_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        line = os.path.splitext(line)[0]
                        ids.add(_strip_known_suffix(line))

        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = _parse_annotation(sample["xml_path"])

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


class ResNetRegressor(nn.Module):
    """
    Backbone ResNet pré-treinada (ImageNet) + head MLP de regressão,
    substituindo inteiramente o DeepLabv3+ (sem saída de segmentação).
    """

    def __init__(self, backbone_name='resnet50', num_outputs=4, pretrained=True):
        super().__init__()

        backbone_fn = getattr(models, backbone_name)

        if pretrained:
            weights_enum = getattr(models, f"{self._weights_name(backbone_name)}_Weights")
            backbone = backbone_fn(weights=weights_enum.DEFAULT)
        else:
            backbone = backbone_fn(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),
        )

    @staticmethod
    def _weights_name(backbone_name: str) -> str:
        # 'resnet50' -> 'ResNet50', 'resnet18' -> 'ResNet18', ...
        return backbone_name.replace("resnet", "ResNet")

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def backbone_params(self):
        return self.backbone.parameters()

    def head_params(self):
        return self.head.parameters()

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable


def calculate_measure_metrics(pred, target, eps=1e-8):
    error = pred - target

    mae = error.abs().mean().item()
    rmse = torch.sqrt((error ** 2).mean()).item()
    rer = (error.abs() / (target.abs() + eps) * 100.0).mean().item()

    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / (ss_tot + eps)).item()

    return mae, rmse, rer, r2


def weighted_measures_loss(pred, target, weights, loss_type='mse', huber_beta=1.0):
    if loss_type == 'mse':
        per_component = ((pred - target) ** 2).mean(dim=0)
    elif loss_type == 'huber':
        diff = (pred - target).abs()
        per_sample = torch.where(
            diff < huber_beta,
            0.5 * diff ** 2 / huber_beta,
            diff - 0.5 * huber_beta,
        )
        per_component = per_sample.mean(dim=0)
    else:
        raise ValueError(f"loss_type desconhecido: {loss_type}")

    return (per_component * weights).sum(), per_component.detach()


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

    train_dataset = LSIDBeansMeasuresDataset(root_dir=ROOT, split_file=TRAIN_CSV, augment=True)
    test_dataset = LSIDBeansMeasuresDataset(root_dir=ROOT, split_file=TEST_CSV, augment=False)

    print(f"Imagens de treino: {len(train_dataset)}")
    print(f"Imagens de validação: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetRegressor(
        backbone_name=BACKBONE,
        num_outputs=NUM_OUTPUTS,
        pretrained=PRETRAINED,
    ).to(device)

    # Congela o backbone no início: treina só a head por algumas validações
    # antes de liberar fine-tuning completo (evita destruir os pesos do
    # ImageNet logo de cara com LR alta).
    model.set_backbone_trainable(False)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.head_params(), "lr": LR_HEAD},
            {"params": model.backbone_params(), "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_HEAD, LR_BACKBONE],
        total_steps=ITERS,
        pct_start=0.1,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    measure_weights = torch.tensor(MEASURE_WEIGHTS_VALUES, device=device)

    best_val_loss = float("inf")
    iteration = 0
    checkpoint_path = "best_model_resnet_measures_resnet.pth"
    unfrozen = False

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint encontrado: {checkpoint_path}")
        print("Carregando checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        iteration = checkpoint["iteration"]
        best_val_loss = checkpoint["val_loss"]
        unfrozen = checkpoint.get("unfrozen", False)
        model.set_backbone_trainable(unfrozen)

        print("Checkpoint carregado!")
        print(f"Continuando da iteração: {iteration}")
        print(f"Melhor validation loss: {best_val_loss:.4f}")
    else:
        print("Nenhum checkpoint encontrado.")
        print("Iniciando treinamento do zero.")

    print(f"Começando [{datetime.now().strftime('%H:%M:%S')}]")

    val_count = 0

    while iteration < ITERS:
        model.train()

        for batch in train_loader:
            if iteration >= ITERS:
                break

            images = batch["image"].to(device, non_blocking=True)

            length = batch["leaf_length"].to(device, non_blocking=True)
            width = batch["leaf_width"].to(device, non_blocking=True)
            perimeter = batch["leaf_perimeter"].to(device, non_blocking=True)
            area = batch["leaf_area"].to(device, non_blocking=True)

            length = (length - LENGTH_MIN) / (LENGTH_MAX - LENGTH_MIN)
            width = (width - WIDTH_MIN) / (WIDTH_MAX - WIDTH_MIN)
            perimeter = (perimeter - PERIMETER_MIN) / (PERIMETER_MAX - PERIMETER_MIN)
            area = (area - AREA_MIN) / (AREA_MAX - AREA_MIN)

            measures_target = torch.stack([length, width, perimeter, area], dim=1)

            optimizer.zero_grad()

            measures_out = model(images)

            if torch.isnan(measures_out).any() or torch.isinf(measures_out).any():
                print(
                    f"[iter {iteration}] measures_out tem NaN/Inf! "
                    f"min={measures_out.min().item()}, max={measures_out.max().item()}"
                )

            loss, per_component = weighted_measures_loss(
                measures_out,
                measures_target,
                measure_weights,
                loss_type=REGRESSION_LOSS,
                huber_beta=HUBER_BETA,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            iteration += 1

            if iteration % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Iteration [{iteration}/{ITERS}] "
                    f"LR: {current_lr:.8f} "
                    f"Loss: {loss.item():.4f} "
                    f"(L={per_component[0]:.4f} W={per_component[1]:.4f} "
                    f"P={per_component[2]:.4f} A={per_component[3]:.4f})"
                )

            if iteration % VAL_INTERVAL == 0:
                val_count += 1

                if not unfrozen and val_count >= FREEZE_BACKBONE_EPOCHS:
                    print(">>> Liberando fine-tuning do backbone (unfreeze).")
                    model.set_backbone_trainable(True)
                    unfrozen = True

                model.eval()

                val_length_preds, val_length_targets = [], []
                val_width_preds, val_width_targets = [], []
                val_perimeter_preds, val_perimeter_targets = [], []
                val_area_preds, val_area_targets = [], []

                val_total_loss = 0.0

                with torch.no_grad():
                    for val_batch in test_loader:
                        val_images = val_batch["image"].to(device, non_blocking=True)

                        val_length = (
                            val_batch["leaf_length"].to(device) - LENGTH_MIN
                        ) / (LENGTH_MAX - LENGTH_MIN)

                        val_width = (
                            val_batch["leaf_width"].to(device) - WIDTH_MIN
                        ) / (WIDTH_MAX - WIDTH_MIN)

                        val_perimeter = (
                            val_batch["leaf_perimeter"].to(device) - PERIMETER_MIN
                        ) / (PERIMETER_MAX - PERIMETER_MIN)

                        val_area = (
                            val_batch["leaf_area"].to(device) - AREA_MIN
                        ) / (AREA_MAX - AREA_MIN)

                        val_measures_target = torch.stack(
                            [val_length, val_width, val_perimeter, val_area], dim=1
                        )

                        val_measures_out = model(val_images)

                        val_loss, _ = weighted_measures_loss(
                            val_measures_out,
                            val_measures_target,
                            measure_weights,
                            loss_type=REGRESSION_LOSS,
                            huber_beta=HUBER_BETA,
                        )

                        val_length_preds.extend(val_measures_out[:, 0].detach().cpu().tolist())
                        val_length_targets.extend(val_measures_target[:, 0].detach().cpu().tolist())

                        val_width_preds.extend(val_measures_out[:, 1].detach().cpu().tolist())
                        val_width_targets.extend(val_measures_target[:, 1].detach().cpu().tolist())

                        val_perimeter_preds.extend(val_measures_out[:, 2].detach().cpu().tolist())
                        val_perimeter_targets.extend(val_measures_target[:, 2].detach().cpu().tolist())

                        val_area_preds.extend(val_measures_out[:, 3].detach().cpu().tolist())
                        val_area_targets.extend(val_measures_target[:, 3].detach().cpu().tolist())

                        val_total_loss += val_loss.item()

                val_total_loss /= len(test_loader)

                val_length_preds = torch.tensor(val_length_preds) * (LENGTH_MAX - LENGTH_MIN) + LENGTH_MIN
                val_length_targets = torch.tensor(val_length_targets) * (LENGTH_MAX - LENGTH_MIN) + LENGTH_MIN

                val_width_preds = torch.tensor(val_width_preds) * (WIDTH_MAX - WIDTH_MIN) + WIDTH_MIN
                val_width_targets = torch.tensor(val_width_targets) * (WIDTH_MAX - WIDTH_MIN) + WIDTH_MIN

                val_perimeter_preds = (
                    torch.tensor(val_perimeter_preds) * (PERIMETER_MAX - PERIMETER_MIN) + PERIMETER_MIN
                )
                val_perimeter_targets = (
                    torch.tensor(val_perimeter_targets) * (PERIMETER_MAX - PERIMETER_MIN) + PERIMETER_MIN
                )

                val_area_preds = torch.tensor(val_area_preds) * (AREA_MAX - AREA_MIN) + AREA_MIN
                val_area_targets = torch.tensor(val_area_targets) * (AREA_MAX - AREA_MIN) + AREA_MIN

                length_mae, length_rmse, length_rer, length_r2 = calculate_measure_metrics(
                    val_length_preds, val_length_targets
                )
                width_mae, width_rmse, width_rer, width_r2 = calculate_measure_metrics(
                    val_width_preds, val_width_targets
                )
                perimeter_mae, perimeter_rmse, perimeter_rer, perimeter_r2 = calculate_measure_metrics(
                    val_perimeter_preds, val_perimeter_targets
                )
                area_mae, area_rmse, area_rer, area_r2 = calculate_measure_metrics(
                    val_area_preds, val_area_targets
                )

                current_lr = optimizer.param_groups[0]["lr"]

                print("\n" + "=" * 60)
                print(f"Iteration [{iteration}/{ITERS}]")
                print(f"LR: {current_lr:.8f}")
                print(f"Validation Loss: {val_total_loss:.4f}")
                print()
                print("  Val Measures:")
                print(
                    f"    Length:    MAE={length_mae:.4f}  RMSE={length_rmse:.4f}  "
                    f"RER={length_rer:.2f}%  R²={length_r2:.4f}"
                )
                print(
                    f"    Width:     MAE={width_mae:.4f}  RMSE={width_rmse:.4f}  "
                    f"RER={width_rer:.2f}%  R²={width_r2:.4f}"
                )
                print(
                    f"    Perimeter: MAE={perimeter_mae:.4f}  RMSE={perimeter_rmse:.4f}  "
                    f"RER={perimeter_rer:.2f}%  R²={perimeter_r2:.4f}"
                )
                print(
                    f"    Area:      MAE={area_mae:.4f}  RMSE={area_rmse:.4f}  "
                    f"RER={area_rer:.2f}%  R²={area_r2:.4f}"
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
                            "length_mae": length_mae,
                            "width_mae": width_mae,
                            "perimeter_mae": perimeter_mae,
                            "area_mae": area_mae,
                            "unfrozen": unfrozen,
                        },
                        checkpoint_path,
                    )

                    print(">>> Melhor modelo salvo!")

                print()

    print("\nTreinamento finalizado.")
    print(f"Melhor validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()