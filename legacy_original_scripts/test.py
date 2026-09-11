import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from leaf_deeplab import (
    LSIDBeansDataset,
    NUM_CLASSES,
    BACKBONE,
    OUTPUT_STRIDE,
    SYNC_BN,
    FREEZE_BN,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    BETA,
    LENGTH_MIN,
    LENGTH_MAX,
    WIDTH_MIN,
    WIDTH_MAX,
    PERIMETER_MIN,
    PERIMETER_MAX,
)

from models.deeplab import DeepLab


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = r"Leaf on Stem Image Dataset Beans (LSID-Beans)\LSID-Beans"

TEST_CSV = (
    r"Leaf on Stem Image Dataset Beans (LSID-Beans)"
    r"\cross-validation\cv_1\split_1\test.csv"
)

CHECKPOINT_PATH = "best_model.pth"

RESULTS_DIR = "test_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Ajuste conforme a normalização usada no seu dataset (leaf_deeplab.py)
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])


# ============================================================
# MÉTRICAS
# ============================================================

def regression_metrics(pred, target, eps=1e-8):
    pred = pred.float()
    target = target.float()

    error = pred - target

    mae = error.abs().mean().item()
    rmse = torch.sqrt((error ** 2).mean()).item()

    rer = (
        error.abs()
        / (target.abs() + eps)
        * 100.0
    ).mean().item()

    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()

    r2 = (
        1.0 - ss_res / (ss_tot + eps)
    ).item()

    pred_c = pred - pred.mean()
    target_c = target - target.mean()

    pearson_r = (
        (pred_c * target_c).sum()
        /
        torch.sqrt(
            (pred_c ** 2).sum()
            * (target_c ** 2).sum()
            + eps
        )
    ).item()

    bias = error.mean().item()

    return {
        "mae": mae,
        "rmse": rmse,
        "rer": rer,
        "r2": r2,
        "pearson_r": pearson_r,
        "bias": bias,
    }


def area_metrics(pred, target, eps=1e-8):
    return regression_metrics(pred, target, eps)


# ============================================================
# FUNÇÕES DE PLOT
# ============================================================

def savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {path}")


def plot_pred_vs_real(pred, real, name, title, unit=""):
    pred = pred.numpy()
    real = real.numpy()

    mn = min(real.min(), pred.min())
    mx = max(real.max(), pred.max())

    plt.figure(figsize=(7, 6))

    plt.scatter(real, pred, s=14, alpha=0.55)

    plt.plot(
        [mn, mx],
        [mn, mx],
        linestyle="--",
        linewidth=2,
        label="Ideal: y = x"
    )

    plt.xlabel(f"Valor real {unit}".strip())
    plt.ylabel(f"Valor predito {unit}".strip())
    plt.title(title)
    plt.legend()

    savefig(name)


def plot_residuals(pred, real, name, title, unit=""):
    pred = pred.numpy()
    real = real.numpy()

    residual = pred - real

    plt.figure(figsize=(7, 6))

    plt.scatter(
        real,
        residual,
        s=14,
        alpha=0.55
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(f"Valor real {unit}".strip())
    plt.ylabel(f"Resíduo (Predito - Real) {unit}".strip())
    plt.title(title)

    savefig(name)


def plot_error_histogram(pred, real, name, title, unit=""):
    error = (
        pred - real
    ).numpy()

    plt.figure(figsize=(7, 5))

    plt.hist(
        error,
        bins=30
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(f"Erro (Predito - Real) {unit}".strip())
    plt.ylabel("Frequência")
    plt.title(title)

    savefig(name)


def plot_rer_distribution(pred, real, name, title):
    pred = pred.numpy()
    real = real.numpy()

    rer = (
        np.abs(pred - real)
        / (np.abs(real) + 1e-8)
        * 100
    )

    # Limita somente a visualização a 50%.
    # Os cálculos das métricas continuam usando todos os exemplos.
    rer_plot = rer[rer <= 50]

    plt.figure(figsize=(7, 5))

    plt.hist(
        rer_plot,
        bins=30
    )

    plt.axvline(
        rer.mean(),
        linestyle="--",
        linewidth=2,
        label=f"Média = {rer.mean():.2f}%"
    )

    plt.xlabel("Erro relativo absoluto (%)")
    plt.ylabel("Frequência")
    plt.title(title)
    plt.legend()

    savefig(name)


def plot_metric_bars(names, values, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    plt.bar(names, values)

    plt.ylabel(ylabel)
    plt.title(title)

    savefig(filename)


def plot_boxplot(errors, labels, title, filename):
    plt.figure(figsize=(8, 5))

    plt.boxplot(
        errors,
        labels=labels,
        showfliers=True
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1.5
    )

    plt.ylabel("Erro (Predito - Real)")
    plt.title(title)

    savefig(filename)





def plot_sample_overview(image, seg_gt, seg_pred, area_map, name, title):
    """
    Plota, lado a lado: imagem original, máscara de segmentação real,
    máscara de segmentação predita e o mapeamento de área predito.
    """
    # 1. Converter imagem de Tensor PyTorch para NumPy e ajustar canais para [H, W, C]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Se a imagem estiver no formato [C, H, W], transpõe para [H, W, C]
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
        
        # Desnormalizar a imagem para exibição correta (usando IMG_MEAN e IMG_STD definidos no topo)
        if image.max() > 1.0:
            image = image / 255.0
        image = np.clip(image, 0, 1)
        image = np.clip(image, 0, 1) # Garante valores válidos entre 0 e 1

    # 2. Converter o restante dos tensores para NumPy
    seg_gt = seg_gt.cpu().numpy()
    seg_pred = seg_pred.cpu().numpy()
    area_map = area_map.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("Imagem original")
    axes[0].axis("off")

    axes[1].imshow(seg_gt, cmap="viridis", vmin=0, vmax=NUM_CLASSES - 1)
    axes[1].set_title("Máscara real")
    axes[1].axis("off")

    axes[2].imshow(seg_pred, cmap="viridis", vmin=0, vmax=NUM_CLASSES - 1)
    axes[2].set_title("Máscara predita")
    axes[2].axis("off")

    im = axes[3].imshow(area_map, cmap="inferno")
    axes[3].set_title("Mapeamento de área predito")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    savefig(name)


def plot_random_test_sample(model, test_dataset, device, idx=None):
    """
    Seleciona uma amostra (aleatória por padrão) do conjunto de teste,
    roda a inferência e plota imagem original / máscara real / máscara
    predita / mapeamento de área.
    """
    if idx is None:
        idx = random.randint(0, len(test_dataset) - 1)

    sample = test_dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)
    seg_gt = sample["seg_mask"]

    with torch.no_grad():
        seg_out, _, area_out, _ = model(image)

    seg_pred = torch.argmax(seg_out, dim=1)[0]

    leaf_mask_pred = (seg_pred == 1).float()
    marker_mask_pred = (seg_pred == 2).float()

    O_f = area_out[0, 0]
    O_m = area_out[0, 1]

    area_map = O_f * leaf_mask_pred + O_m * marker_mask_pred

    plot_sample_overview(
        image[0],
        seg_gt,
        seg_pred,
        area_map,
        "22_random_sample_overview.png",
        f"Amostra de teste #{idx}"
    )


# ============================================================
# TESTE
# ============================================================

def main():

    print("=" * 70)
    print("AVALIAÇÃO FINAL DO MELHOR MODELO")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    test_dataset = LSIDBeansDataset(
        root_dir=ROOT,
        split_file=TEST_CSV,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print(
        f"Imagens de teste: {len(test_dataset)}"
    )

    print()

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    model = DeepLab(
        num_classes=NUM_CLASSES,
        backbone=BACKBONE,
        output_stride=OUTPUT_STRIDE,
        sync_bn=SYNC_BN,
        freeze_bn=FREEZE_BN,
    ).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {CHECKPOINT_PATH}"
        )

    print(f"Carregando: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint da iteração: "
        f"{checkpoint.get('iteration', 'N/A')}"
    )

    print(
        f"Validation loss do checkpoint: "
        f"{checkpoint.get('val_loss', float('nan')):.6f}"
    )

    print()

    criterion_seg = nn.CrossEntropyLoss()
    criterion_area = nn.MSELoss()
    criterion_measures = nn.MSELoss()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_area_loss = 0.0
    total_measures_loss = 0.0
    num_batches = 0

    # --------------------------------------------------------
    # ARMAZENAMENTO
    # --------------------------------------------------------

    length_preds = []
    length_targets = []

    width_preds = []
    width_targets = []

    perimeter_preds = []
    perimeter_targets = []

    leaf_area_preds = []
    leaf_area_targets = []

    marker_area_preds = []
    marker_area_targets = []

    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64
    )

    # --------------------------------------------------------
    # INFERÊNCIA
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            seg_mask = batch["seg_mask"].to(
                device,
                non_blocking=True
            )

            area_mask = batch["area_mask"].to(
                device,
                non_blocking=True
            )

            # -------------------------
            # TARGETS DAS MEASURES
            # -------------------------

            length = (
                batch["leaf_length"].to(device)
                - LENGTH_MIN
            ) / (
                LENGTH_MAX - LENGTH_MIN
            )

            width = (
                batch["leaf_width"].to(device)
                - WIDTH_MIN
            ) / (
                WIDTH_MAX - WIDTH_MIN
            )

            perimeter = (
                batch["leaf_perimeter"].to(device)
                - PERIMETER_MIN
            ) / (
                PERIMETER_MAX - PERIMETER_MIN
            )

            measures_target = torch.stack(
                [
                    length,
                    width,
                    perimeter
                ],
                dim=1
            )

            # -------------------------
            # MODELO
            # -------------------------

            seg_out, _, area_out, measures_out = model(
                images
            )

            # -------------------------
            # SEGMENTAÇÃO
            # -------------------------

            loss_seg = criterion_seg(
                seg_out,
                seg_mask
            )

            seg_pred = torch.argmax(
                seg_out,
                dim=1
            )

            # -------------------------
            # ÁREA
            # -------------------------

            leaf_mask_pred = (
                seg_pred == 1
            ).float().unsqueeze(1)

            marker_mask_pred = (
                seg_pred == 2
            ).float().unsqueeze(1)

            leaf_mask_gt = (
                seg_mask == 1
            ).float().unsqueeze(1)

            marker_mask_gt = (
                seg_mask == 2
            ).float().unsqueeze(1)

            O_f = area_out[:, 0:1]
            O_m = area_out[:, 1:2]

            A_f = O_f * leaf_mask_pred
            A_m = O_m * marker_mask_pred

            area_gt = area_mask[:, 0:1]

            E_f = (
                area_gt
                * leaf_mask_gt
                * BETA
            )

            E_m = (
                area_gt
                * marker_mask_gt
                * BETA
            )

            loss_area = (
                criterion_area(A_f, E_f)
                +
                criterion_area(A_m, E_m)
            )

            # -------------------------
            # MEASURES
            # -------------------------

            loss_measures = criterion_measures(
                measures_out,
                measures_target
            )

            loss = (
                loss_seg
                + loss_area
                + loss_measures
            )

            total_loss += loss.item()
            total_seg_loss += loss_seg.item()
            total_area_loss += loss_area.item()
            total_measures_loss += loss_measures.item()
            num_batches += 1

            # -------------------------
            # CONFUSION MATRIX
            # -------------------------

            pred_flat = seg_pred.reshape(-1).cpu()
            target_flat = seg_mask.reshape(-1).cpu()

            for cls in range(NUM_CLASSES):

                target_cls = target_flat == cls

                for pred_cls in range(NUM_CLASSES):

                    confusion[cls, pred_cls] += (
                        target_cls
                        & (pred_flat == pred_cls)
                    ).sum()

            # -------------------------
            # MEASURES
            # -------------------------

            length_preds.append(
                measures_out[:, 0].cpu()
            )
            length_targets.append(
                measures_target[:, 0].cpu()
            )

            width_preds.append(
                measures_out[:, 1].cpu()
            )
            width_targets.append(
                measures_target[:, 1].cpu()
            )

            perimeter_preds.append(
                measures_out[:, 2].cpu()
            )
            perimeter_targets.append(
                measures_target[:, 2].cpu()
            )

            # -------------------------
            # ÁREA POR IMAGEM
            # -------------------------

            pred_leaf_area = (
                A_f.sum(dim=(1, 2, 3))
                / BETA
            )

            real_leaf_area = (
                E_f.sum(dim=(1, 2, 3))
            )

            pred_marker_area = (
                A_m.sum(dim=(1, 2, 3))
                / BETA
            )

            real_marker_area = (
                E_m.sum(dim=(1, 2, 3))
            )

            leaf_area_preds.append(
                pred_leaf_area.cpu()
            )
            leaf_area_targets.append(
                real_leaf_area.cpu()
            )

            marker_area_preds.append(
                pred_marker_area.cpu()
            )
            marker_area_targets.append(
                real_marker_area.cpu()
            )

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Processado: "
                    f"{batch_idx + 1}/"
                    f"{len(test_loader)} batches"
                )

    # ========================================================
    # CONCATENAR
    # ========================================================

    length_preds = torch.cat(length_preds)
    length_targets = torch.cat(length_targets)

    width_preds = torch.cat(width_preds)
    width_targets = torch.cat(width_targets)

    perimeter_preds = torch.cat(perimeter_preds)
    perimeter_targets = torch.cat(perimeter_targets)

    leaf_area_preds = torch.cat(leaf_area_preds)
    leaf_area_targets = torch.cat(leaf_area_targets)

    marker_area_preds = torch.cat(marker_area_preds)
    marker_area_targets = torch.cat(marker_area_targets)

    # ========================================================
    # DESNORMALIZAÇÃO
    # ========================================================

    length_preds = (
        length_preds
        * (LENGTH_MAX - LENGTH_MIN)
        + LENGTH_MIN
    )

    length_targets = (
        length_targets
        * (LENGTH_MAX - LENGTH_MIN)
        + LENGTH_MIN
    )

    width_preds = (
        width_preds
        * (WIDTH_MAX - WIDTH_MIN)
        + WIDTH_MIN
    )

    width_targets = (
        width_targets
        * (WIDTH_MAX - WIDTH_MIN)
        + WIDTH_MIN
    )

    perimeter_preds = (
        perimeter_preds
        * (PERIMETER_MAX - PERIMETER_MIN)
        + PERIMETER_MIN
    )

    perimeter_targets = (
        perimeter_targets
        * (PERIMETER_MAX - PERIMETER_MIN)
        + PERIMETER_MIN
    )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    length_m = regression_metrics(
        length_preds,
        length_targets
    )

    width_m = regression_metrics(
        width_preds,
        width_targets
    )

    perimeter_m = regression_metrics(
        perimeter_preds,
        perimeter_targets
    )

    leaf_area_m = area_metrics(
        leaf_area_preds,
        leaf_area_targets
    )

    marker_area_m = area_metrics(
        marker_area_preds,
        marker_area_targets
    )

    # ========================================================
    # SEGMENTAÇÃO
    # ========================================================

    intersection = torch.diag(confusion).float()
    target_pixels = confusion.sum(dim=1).float()
    pred_pixels = confusion.sum(dim=0).float()

    union = (
        target_pixels
        + pred_pixels
        - intersection
    )

    iou = intersection / (union + 1e-8)

    dice = (
        2 * intersection
        / (
            target_pixels
            + pred_pixels
            + 1e-8
        )
    )

    miou = iou[union > 0].mean().item()

    mean_dice = dice[
        (target_pixels + pred_pixels) > 0
    ].mean().item()

    pixel_accuracy = (
        intersection.sum()
        / (confusion.sum() + 1e-8)
    ).item()

    # ========================================================
    # RESULTADOS NO TERMINAL
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADO FINAL NO CONJUNTO DE TESTE")
    print("=" * 70)

    print()
    print("LOSSES")
    print(f"  Total:        {total_loss / num_batches:.6f}")
    print(f"  Segmentation: {total_seg_loss / num_batches:.6f}")
    print(f"  Area:         {total_area_loss / num_batches:.6f}")
    print(f"  Measures:     {total_measures_loss / num_batches:.6f}")

    print()
    print("SEGMENTAÇÃO")
    print(f"  Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"  mIoU:           {miou:.4f}")
    print(f"  Mean Dice:      {mean_dice:.4f}")

    names = ["Background", "Leaf", "Marker"]

    for i, name in enumerate(names):
        print(
            f"  {name:<10} "
            f"IoU={iou[i].item():.4f} "
            f"Dice={dice[i].item():.4f}"
        )

    print()
    print("ÁREA")

    for name, m in [
        ("Folha", leaf_area_m),
        ("Marcador", marker_area_m)
    ]:
        print(f"  {name}:")
        print(f"    MAE:  {m['mae']:.4f}")
        print(f"    RMSE: {m['rmse']:.4f}")
        print(f"    RER:  {m['rer']:.4f}%")
        print(f"    R²:   {m['r2']:.4f}")
        print(f"    Bias: {m['bias']:+.4f}")

    print()
    print("MEASURES")

    for name, m in [
        ("Length", length_m),
        ("Width", width_m),
        ("Perimeter", perimeter_m)
    ]:
        print(f"  {name}:")
        print(f"    MAE:       {m['mae']:.4f}")
        print(f"    RMSE:      {m['rmse']:.4f}")
        print(f"    RER:       {m['rer']:.4f}%")
        print(f"    R²:        {m['r2']:.4f}")
        print(f"    Pearson r: {m['pearson_r']:.4f}")
        print(f"    Bias:      {m['bias']:+.4f}")

    # ========================================================
    # GRÁFICOS
    # ========================================================

    print()
    print("=" * 70)
    print("GERANDO GRÁFICOS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. PREDITO X REAL — MEASURES
    # --------------------------------------------------------

    plot_pred_vs_real(
        length_preds,
        length_targets,
        "01_length_pred_vs_real.png",
        "Length — Predito vs. Real"
    )

    plot_pred_vs_real(
        width_preds,
        width_targets,
        "02_width_pred_vs_real.png",
        "Width — Predito vs. Real"
    )

    plot_pred_vs_real(
        perimeter_preds,
        perimeter_targets,
        "03_perimeter_pred_vs_real.png",
        "Perimeter — Predito vs. Real"
    )

    # --------------------------------------------------------
    # 2. PREDITO X REAL — ÁREA
    # --------------------------------------------------------

    plot_pred_vs_real(
        leaf_area_preds,
        leaf_area_targets,
        "04_leaf_area_pred_vs_real.png",
        "Área da Folha — Predito vs. Real"
    )

    plot_pred_vs_real(
        marker_area_preds,
        marker_area_targets,
        "05_marker_area_pred_vs_real.png",
        "Área do Marcador — Predito vs. Real"
    )

    # --------------------------------------------------------
    # 3. RESÍDUOS × REAL
    # --------------------------------------------------------

    plot_residuals(
        length_preds,
        length_targets,
        "06_length_residuals.png",
        "Length — Resíduos vs. Valor Real"
    )

    plot_residuals(
        width_preds,
        width_targets,
        "07_width_residuals.png",
        "Width — Resíduos vs. Valor Real"
    )

    plot_residuals(
        perimeter_preds,
        perimeter_targets,
        "08_perimeter_residuals.png",
        "Perimeter — Resíduos vs. Valor Real"
    )

    plot_residuals(
        leaf_area_preds,
        leaf_area_targets,
        "09_leaf_area_residuals.png",
        "Área da Folha — Resíduos vs. Valor Real"
    )

    plot_residuals(
        marker_area_preds,
        marker_area_targets,
        "10_marker_area_residuals.png",
        "Área do Marcador — Resíduos vs. Valor Real"
    )

    # --------------------------------------------------------
    # 4. HISTOGRAMAS DOS ERROS
    # --------------------------------------------------------

    plot_error_histogram(
        length_preds,
        length_targets,
        "11_length_error_histogram.png",
        "Length — Distribuição dos Erros"
    )

    plot_error_histogram(
        width_preds,
        width_targets,
        "12_width_error_histogram.png",
        "Width — Distribuição dos Erros"
    )

    plot_error_histogram(
        perimeter_preds,
        perimeter_targets,
        "13_perimeter_error_histogram.png",
        "Perimeter — Distribuição dos Erros"
    )

    # --------------------------------------------------------
    # 5. DISTRIBUIÇÃO DO RER
    # --------------------------------------------------------

    plot_rer_distribution(
        length_preds,
        length_targets,
        "14_length_rer_distribution.png",
        "Length — Distribuição do RER"
    )

    plot_rer_distribution(
        width_preds,
        width_targets,
        "15_width_rer_distribution.png",
        "Width — Distribuição do RER"
    )

    plot_rer_distribution(
        perimeter_preds,
        perimeter_targets,
        "16_perimeter_rer_distribution.png",
        "Perimeter — Distribuição do RER"
    )

    # --------------------------------------------------------
    # 6. IoU E DICE POR CLASSE
    # --------------------------------------------------------

    plot_metric_bars(
        names,
        iou.numpy() * 100,
        "IoU (%)",
        "IoU por Classe",
        "17_iou_by_class.png"
    )

    plot_metric_bars(
        names,
        dice.numpy() * 100,
        "Dice (%)",
        "Dice por Classe",
        "18_dice_by_class.png"
    )

    # --------------------------------------------------------
    # 7. RER DAS MEASURES
    # --------------------------------------------------------

    measure_names = [
        "Length",
        "Width",
        "Perimeter"
    ]

    measure_rer = [
        length_m["rer"],
        width_m["rer"],
        perimeter_m["rer"]
    ]

    plot_metric_bars(
        measure_names,
        measure_rer,
        "RER (%)",
        "RER das Measures",
        "19_measures_rer.png"
    )

    # --------------------------------------------------------
    # 8. R² DAS MEASURES
    # --------------------------------------------------------

    measure_r2 = [
        length_m["r2"],
        width_m["r2"],
        perimeter_m["r2"]
    ]

    plot_metric_bars(
        measure_names,
        measure_r2,
        "R²",
        "R² das Measures",
        "20_measures_r2.png"
    )

    # --------------------------------------------------------
    # 9. BOX PLOT DOS ERROS
    # --------------------------------------------------------

    errors = [
        (
            length_preds - length_targets
        ).numpy(),

        (
            width_preds - width_targets
        ).numpy(),

        (
            perimeter_preds - perimeter_targets
        ).numpy()
    ]

    plot_boxplot(
        errors,
        measure_names,
        "Distribuição dos Erros — Measures",
        "21_measures_error_boxplot.png"
    )

    # --------------------------------------------------------
    # 10. AMOSTRA ALEATÓRIA — IMAGEM / SEGMENTAÇÃO / ÁREA
    # --------------------------------------------------------

    plot_random_test_sample(model, test_dataset, device)

    print()
    print("=" * 70)
    print(
        f"Todos os gráficos foram salvos em: "
        f"{RESULTS_DIR}/"
    )
    print("=" * 70)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()