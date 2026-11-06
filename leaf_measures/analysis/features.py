"""
Características de cada imagem que PODEM explicar o erro do modelo.

Tudo é calculado a partir dos dados (anotação XML, imagem, máscara de
segmentação), sem rodar modelo nenhum. O resultado é salvo em cache
(results/analise/features.csv) porque não depende do experimento.

Para adicionar uma característica nova: calcule-a em `_image_features` ou
`_mask_features` e descreva-a em FEATURE_INFO (o texto aparece nos indícios).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image

from leaf_measures.data.annotation import parse_annotation
from leaf_measures.data.datasets import LEAF_ID, MARKER_ID, SEG_DTYPE, index_samples

# nome -> (descrição curta, unidade). Só as características listadas aqui entram na busca de padrões.
FEATURE_INFO = {
    "leaf_area": ("área real da folha (anotada)", "cm²"),
    "leaf_length": ("comprimento real da folha (anotado)", "cm"),
    "leaf_width": ("largura real da folha (anotada)", "cm"),
    "leaf_perimeter": ("perímetro real da folha (anotado)", "cm"),
    "marker_px": ("área do marcador na foto", "px"),
    "px_per_cm2": ("escala da foto (pixels por cm², via marcador)", "px/cm²"),
    "leaf_frac": ("fração da imagem ocupada pela folha", "%"),
    "marker_frac": ("fração da imagem ocupada pelo marcador", "%"),
    "leaf_touches_border": ("folha encostando na borda da imagem", "sim/não"),
    "marker_touches_border": ("marcador encostando na borda da imagem", "sim/não"),
    "leaf_bbox_aspect": ("alongamento da folha na foto (lado maior / menor da caixa)", "razão"),
    "leaf_fill": ("compacidade da folha (pixels da folha / caixa envolvente)", "razão"),
    "geom_area_rer": ("divergência entre a área geométrica (máscara + marcador) e a área anotada", "%"),
    "brightness": ("brilho médio da imagem", "0–255"),
    "contrast": ("contraste da imagem (desvio padrão do cinza)", "0–255"),
    "sharpness": ("nitidez da imagem (variância do Laplaciano)", "u.a."),
    "leaf_brightness": ("brilho médio da folha", "0–255"),
    "saturation": ("saturação média das cores", "0–1"),
    "has_observation": ("anotação com campo 'observation' preenchido", "sim/não"),
}
BINARY_FEATURES = {"leaf_touches_border", "marker_touches_border", "has_observation"}
CATEGORICAL_FEATURES = {"leaf_folder"}


def feature_label(name: str) -> str:
    return FEATURE_INFO.get(name, (name, ""))[0]


def compute_features(root_dir: str, height: int = 512, width: int = 512, verbose: bool = True) -> pd.DataFrame:
    """Uma linha por imagem do dataset, indexada por 'name' = '<pasta_folha>/<image-name do XML>'."""
    samples = index_samples(root_dir, require_seg=False)
    rows = []
    for i, s in enumerate(samples):
        if verbose and i % 200 == 0:
            print(f"  características: {i}/{len(samples)}")
        ann = parse_annotation(s["xml_path"])
        row = {
            "name": f"{s['leaf_folder']}/{ann['image_name']}",
            "sample_id": s["id"],
            "leaf_folder": s["leaf_folder"],
            "image_path": s["image_path"],
            "leaf_area": ann["leaf_area"],
            "leaf_length": ann["leaf_length"],
            "leaf_width": ann["leaf_width"],
            "leaf_perimeter": ann["leaf_perimeter"],
            "marker_real_area": ann["marker_real_area"],
            "marker_px": ann["marker_projected_area"],
            "px_per_cm2": ann["marker_projected_area"] / ann["marker_real_area"] if ann["marker_real_area"] else np.nan,
            "observation": (ann["observation"] or "").strip(),
        }
        row["has_observation"] = float(bool(row["observation"]))

        gray, rgb = _load_image(s["image_path"])
        row.update(_image_features(gray, rgb))

        seg_path = os.path.join(root_dir, s["leaf_folder"], "segmentation", s["id"] + "_segmentation.raw")
        row["seg_path"] = seg_path if os.path.isfile(seg_path) else ""
        if row["seg_path"]:
            seg = _load_seg(seg_path, height, width)
            if seg is not None:
                row.update(_mask_features(seg, gray, ann))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("name")
    if df.index.duplicated().any():
        dup = df.index[df.index.duplicated()].unique()[:5].tolist()
        print(f"[AVISO] nomes de imagem duplicados no dataset (mantendo o primeiro): {dup}")
        df = df[~df.index.duplicated()]
    return df


def load_or_compute_features(cache_path: str, root_dir: str, height: int, width: int, recompute: bool = False):
    if os.path.isfile(cache_path) and not recompute:
        df = pd.read_csv(cache_path, index_col="name", keep_default_na=False, na_values=[""])
        missing = [f for f in FEATURE_INFO if f not in df.columns]
        same_root = df["image_path"].astype(str).str.startswith(root_dir).all() if len(df) else False
        if not missing and same_root:
            print(f"Usando características em cache: {cache_path} (use --recompute-features para refazer)")
            return df
        print("Cache de características desatualizado (dataset ou lista de características mudou); recalculando.")
    print(f"Calculando características das imagens em {root_dir} ...")
    df = compute_features(root_dir, height, width)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_csv(cache_path)
    print(f"Características salvas em: {cache_path}")
    return df


# ------------------------------------------------------------------ internos
def _load_image(path):
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return gray, rgb


def _load_seg(path, height, width):
    arr = np.fromfile(path, dtype=SEG_DTYPE)
    if arr.size != height * width:
        return None
    return arr.reshape(height, width)


def _laplacian_var(gray: np.ndarray) -> float:
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def _image_features(gray, rgb) -> dict:
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": _laplacian_var(gray),
        "saturation": float(sat.mean()),
    }


def _touches_border(mask: np.ndarray) -> float:
    return float(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _mask_features(seg, gray, ann) -> dict:
    leaf = seg == LEAF_ID
    marker = seg == MARKER_ID
    total = seg.size
    leaf_px, marker_px_mask = int(leaf.sum()), int(marker.sum())
    out = {
        "leaf_frac": 100.0 * leaf_px / total,
        "marker_frac": 100.0 * marker_px_mask / total,
        "leaf_touches_border": _touches_border(leaf),
        "marker_touches_border": _touches_border(marker),
        "leaf_brightness": float(gray[leaf].mean()) if leaf_px else np.nan,
    }
    if leaf_px:
        ys, xs = np.nonzero(leaf)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        out["leaf_bbox_aspect"] = float(max(h, w) / max(min(h, w), 1))
        out["leaf_fill"] = float(leaf_px / (h * w))
    else:
        out["leaf_bbox_aspect"] = out["leaf_fill"] = np.nan

    # Área "geométrica": pixels da folha convertidos em cm² pela escala do marcador da própria máscara.
    # Se diverge muito da área anotada, a máscara ou a anotação daquela imagem pode ter problema.
    if leaf_px and marker_px_mask and ann["marker_real_area"]:
        geom = leaf_px / marker_px_mask * ann["marker_real_area"]
        out["geom_area"] = geom
        out["geom_area_rer"] = 100.0 * abs(geom - ann["leaf_area"]) / max(ann["leaf_area"], 1e-8)
    else:
        out["geom_area"] = out["geom_area_rer"] = np.nan
    return out
