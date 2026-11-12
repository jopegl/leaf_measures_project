"""Relatório da análise de erros: CSVs, grades de imagens, gráficos e indicios.md."""
from __future__ import annotations

import os
import shutil
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from leaf_measures.analysis.features import BINARY_FEATURES, feature_label  # noqa: E402
from leaf_measures.analysis.patterns import MEASURE_PT, target_columns  # noqa: E402
from leaf_measures.analysis.ranking import multi_measure_outliers, rank_images  # noqa: E402
from leaf_measures.data.datasets import LEAF_ID, MARKER_ID, SEG_DTYPE  # noqa: E402
from leaf_measures.data.normalization import MEASURE_ORDER  # noqa: E402

DPI = 120
SHORT = {"length": "C", "width": "L", "perimeter": "P", "area": "A"}


# ------------------------------------------------------------------ imagens
def _thumb(row, size=256):
    path = row.get("image_path")
    if not isinstance(path, str) or not os.path.isfile(path):
        return None, None
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    img = img.resize((size, int(size * h0 / w0)))
    seg = None
    seg_path = row.get("seg_path")
    if isinstance(seg_path, str) and seg_path and os.path.isfile(seg_path):
        arr = np.fromfile(seg_path, dtype=SEG_DTYPE)
        if arr.size == w0 * h0:
            seg = np.asarray(Image.fromarray(arr.reshape(h0, w0)).resize(img.size, Image.NEAREST))
    return np.asarray(img), seg


def image_grid(rows: pd.DataFrame, target_key: str, title: str, path: str, ncols: int = 4, caption_fn=None):
    n = len(rows)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, r) in zip(axes, rows.iterrows()):
        img, seg = _thumb(r)
        if img is not None:
            ax.imshow(img)
            if seg is not None:
                ax.contour(seg == LEAF_ID, levels=[0.5], colors="yellow", linewidths=0.8)
                ax.contour(seg == MARKER_ID, levels=[0.5], colors="cyan", linewidths=0.8)
        else:
            ax.text(0.5, 0.5, "imagem não encontrada", ha="center", va="center")
        if caption_fn is not None:
            line2 = caption_fn(r)
        elif target_key == "mean":
            sub = " ".join(f"{SHORT[m]} {r[f'{m}_rer']:.1f}%" for m in MEASURE_ORDER if f"{m}_rer" in r)
            line2 = f"RER médio {r['rer_mean']:.1f}%\n{sub}"
        else:
            m = target_key
            line2 = (f"real {r[f'{m}_real']:.2f} | pred {r[f'{m}_pred']:.2f}\n"
                     f"RER {r[f'{m}_rer']:.1f}% (n folds {int(r.get('n_folds', 1))})")
        ax.set_title(f"{name}\n{line2}", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title + "   (contorno amarelo = folha, ciano = marcador)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ gráficos
def scatter_feature(table, feature, target_col, measure_key, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x, y = table[feature], table[target_col]
    if feature in BINARY_FEATURES:
        groups = [y[x == v].dropna() for v in (0, 1)]
        ax.boxplot(groups, showfliers=True)
        ax.set_xticks([1, 2], ["não", "sim"])
    else:
        ax.scatter(x, y, s=12, alpha=0.5, edgecolor="k", linewidth=0.2)
        ok = x.notna() & y.notna()
        if ok.sum() > 10:  # tendência por faixas (mediana por decil)
            bins = pd.qcut(x[ok], 10, duplicates="drop")
            trend = y[ok].groupby(bins, observed=True).median()
            centers = x[ok].groupby(bins, observed=True).median()
            ax.plot(centers.values, trend.values, "r-o", ms=3, label="mediana por faixa")
            ax.legend(fontsize=8)
    ax.set_xlabel(feature_label(feature))
    ax.set_ylabel(f"RER {MEASURE_PT[measure_key]} (%)")
    ax.set_title(f"Erro × {feature_label(feature)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def bias_plot(table, path):
    ms = [m for m in MEASURE_ORDER if f"{m}_real" in table]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.5 * len(ms), 4))
    for ax, m in zip(np.atleast_1d(axes), ms):
        ax.scatter(table[f"{m}_real"], table[f"{m}_error"], s=10, alpha=0.5, edgecolor="k", linewidth=0.2)
        ax.axhline(0, color="r", ls="--")
        ax.set_xlabel(f"{MEASURE_PT[m]} real")
        ax.set_ylabel("erro (pred − real)")
        ax.set_title(MEASURE_PT[m].capitalize())
    fig.suptitle("Viés: erro com sinal × valor real (acima de 0 = superestima)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def folders_plot(folders_df, global_mean, path, top=20):
    if folders_df.empty:
        return
    df = folders_df.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1.5))
    ax.barh(df.index.astype(str), df["rer_medio"], color="steelblue", edgecolor="black")
    ax.axvline(global_mean, color="r", ls="--", label=f"média geral {global_mean:.2f}%")
    ax.set_xlabel("RER médio (%)")
    ax.set_title(f"Pastas de folha com maior erro (top {len(df)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ relatório
def write_experiment_report(exp, table, clues, stat_tables, out_dir, top_n=12, quantile=0.1, images=True,
                            level="imagem"):
    for sub in ("graficos", "imagens"):  # remove saídas de execuções anteriores
        shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    gdir = os.path.join(out_dir, "graficos")
    os.makedirs(gdir, exist_ok=True)

    table.to_csv(os.path.join(out_dir, "imagens_erros.csv"))
    for name, df in stat_tables.items():
        if not df.empty:
            df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=(name == "pastas"))

    targets = target_columns(table)
    for key, col in targets.items():
        worst = rank_images(table, col, top_n, worst=True)
        best = rank_images(table, col, top_n, worst=False)
        worst.to_csv(os.path.join(out_dir, f"piores_{key}.csv"))
        best.to_csv(os.path.join(out_dir, f"melhores_{key}.csv"))
        if images:
            label = MEASURE_PT[key]
            image_grid(worst, key, f"{exp} — {top_n} PIORES: {label}", os.path.join(out_dir, "imagens", f"piores_{key}.png"))
            image_grid(best, key, f"{exp} — {top_n} MELHORES: {label}", os.path.join(out_dir, "imagens", f"melhores_{key}.png"))

    multi = multi_measure_outliers(table, quantile)
    multi.to_csv(os.path.join(out_dir, "piores_em_varias_medidas.csv"))
    if images and not multi.empty:
        image_grid(multi.head(top_n), "mean", f"{exp} — piores em ≥3 medidas",
                   os.path.join(out_dir, "imagens", "piores_em_varias_medidas.png"))

    if "geom_area_rer" in table:
        sus = table[table["geom_area_rer"] > 15].sort_values("geom_area_rer", ascending=False)
        sus.to_csv(os.path.join(out_dir, "suspeitas_anotacao.csv"))

    bias_plot(table, os.path.join(gdir, "vies_erro_vs_real.png"))
    folders_plot(stat_tables.get("pastas", pd.DataFrame()), table["rer_mean"].mean(), os.path.join(gdir, "pastas.png"))
    plotted = set()
    for c in clues:
        if c.feature and c.kind == "caracteristica" and (c.feature, c.measure) not in plotted:
            plotted.add((c.feature, c.measure))
            scatter_feature(table, c.feature, targets[c.measure], c.measure,
                            os.path.join(gdir, f"erro_{c.measure}_vs_{c.feature}.png"))
        if len(plotted) >= 8:
            break

    _write_markdown(exp, table, clues, multi, out_dir, quantile, top_n, level)


def _write_markdown(exp, table, clues, multi, out_dir, quantile, top_n, level="imagem"):
    pct = int(round(quantile * 100))
    unit = "folhas" if level == "folha" else "imagens"
    L = [f"# Análise de erros — {exp}", "",
         f"_Gerado em {datetime.now():%d/%m/%Y %H:%M} · {len(table)} {unit} de teste · "
         f"erro por {level} = média sobre os folds em que ela foi testada (até {int(table['n_folds'].max())})._", ""]
    if level == "folha":
        L += ["_Análise por FOLHA: cada linha é uma folha; as características são a média das imagens dela "
              "e as grades mostram a primeira imagem de cada folha. Características do tipo sim/não viram "
              "a fração das imagens da folha._", ""]
    L += [
         "## Resumo", ""]
    for m in MEASURE_ORDER:
        if f"{m}_rer" in table:
            r = table[f"{m}_rer"]
            L.append(f"- **{MEASURE_PT[m].capitalize()}**: RER mediano {r.median():.2f}%, "
                     f"90% das {unit} abaixo de {r.quantile(0.9):.2f}%, pior {r.max():.2f}%")
    L += ["", "## Indícios encontrados", ""]
    main = [c for c in clues if c.strength in ("forte", "moderado")]
    if not main:
        L.append("Nenhum padrão forte ou moderado encontrado entre o erro e as características medidas. "
                 "O erro parece espalhado; vale olhar as grades de imagens em busca de algo que não "
                 "foi medido (ex.: folha dobrada, sombra, folha sobreposta ao marcador).")
    for strength in ("forte", "moderado"):
        group = [c for c in main if c.strength == strength]
        if group:
            L.append(f"### {'🔴 Fortes' if strength == 'forte' else '🟡 Moderados'}")
            L.append("")
            L += [f"- **[{c.scope}]** {c.text}" for c in group]
            L.append("")
    info = [c for c in clues if c.strength == "info"]
    if info:
        L += ["### ℹ️ Observações", ""] + [f"- {c.text}" for c in info] + [""]
    if not multi.empty:
        L += [f"### Imagens ruins em várias medidas ao mesmo tempo ({len(multi)})", "",
              f"Estão entre as {pct}% piores em pelo menos 3 das 4 medidas. Erros assim costumam vir "
              "da imagem ou da anotação, não de uma medida específica:", ""]
        L += [f"- `{n}` — RER médio {r['rer_mean']:.1f}% ({r['medidas_piores']})" for n, r in multi.head(10).iterrows()]
        L.append("")
    L += ["## Como ler", "",
          "- **ρ (Spearman)**: de −1 a +1. Perto de 0 = sem relação; |ρ| ≥ 0,3 moderada; |ρ| ≥ 0,5 forte.",
          f"- **P(piores > demais)**: chance de uma imagem das {pct}% piores ter valor maior da característica "
          "do que uma imagem das demais. 0,5 = sem diferença.",
          "- Só são listados achados com efeito ≥ 0,3 e significativos após correção para múltiplos "
          "testes (Benjamini-Hochberg, FDR 5%). Sem essa correção, testar ~200 combinações "
          "produziria alguns 'achados' por puro acaso.",
          "- Com poucas imagens (ex.: 1 ou 2 folds) só efeitos grandes aparecem. Ausência de indício "
          "não prova ausência de padrão.",
          "- São **correlações, não causas**. Confirme olhando as imagens em `imagens/`.", "",
          "## Arquivos", "",
          f"- `imagens/piores_*.png`, `imagens/melhores_*.png`: as {top_n} piores/melhores por medida",
          "- `imagens_erros.csv`: erro de cada imagem + todas as características (abra no Excel/pandas)",
          "- `correlacoes.csv`, `piores_vs_demais.csv`, `pastas.csv`, `vies.csv`: números completos",
          "- `suspeitas_anotacao.csv`: área pela máscara + marcador diverge > 15% da anotada",
          "- `graficos/`: erro × características dos indícios, viés, pastas"]
    with open(os.path.join(out_dir, "indicios.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def write_comparison_report(cross: pd.DataFrame, agree: pd.DataFrame, features: pd.DataFrame,
                            tables: dict, out_dir, top_n=12, quantile=0.1, images=True):
    os.makedirs(out_dir, exist_ok=True)
    exps = list(tables)
    cross.to_csv(os.path.join(out_dir, "erro_por_imagem_entre_modelos.csv"))
    agree.to_csv(os.path.join(out_dir, "concordancia_entre_modelos.csv"))

    all_hard = cross[cross["classificacao"] == "difícil para todos"]
    if images and not all_hard.empty:
        rows = all_hard.head(top_n).join(features[["image_path", "seg_path"]], how="left")
        image_grid(rows, "mean", "Difíceis para TODOS os modelos", os.path.join(out_dir, "dificeis_para_todos.png"),
                   caption_fn=lambda r: "\n".join(f"{e}: RER {r[e]:.1f}%" for e in exps))

    if len(exps) <= 5:
        n = len(exps)
        fig, axes = plt.subplots(n - 1, n - 1, figsize=(4 * (n - 1), 4 * (n - 1)), squeeze=False)
        for i in range(1, n):
            for j in range(n - 1):
                ax = axes[i - 1, j]
                if j >= i:
                    ax.axis("off")
                    continue
                a, b = exps[j], exps[i]
                ax.scatter(cross[a], cross[b], s=8, alpha=0.5)
                lim = [0, max(cross[a].max(), cross[b].max())]
                ax.plot(lim, lim, "r--", lw=0.8)
                ax.set_xlabel(f"RER médio {a}")
                ax.set_ylabel(f"RER médio {b}")
        fig.suptitle("Erro por imagem entre modelos (pontos longe da diagonal = só um modelo erra)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "erro_entre_modelos.png"), dpi=DPI)
        plt.close(fig)

    pct = int(round(quantile * 100))
    counts = cross["classificacao"].value_counts()
    L = ["# Comparação de erros entre modelos", "", f"Modelos: {', '.join(f'`{e}`' for e in exps)} · "
         f"{len(cross)} imagens em comum", "",
         "## Concordância (Spearman do RER médio por imagem)", "",
         "Valores altos = os modelos erram nas mesmas imagens (o problema tende a estar nos dados); "
         "valores baixos = cada modelo tem dificuldades diferentes.", ""]
    L += ["| | " + " | ".join(exps) + " |", "|---" * (len(exps) + 1) + "|"]
    for e in exps:
        L.append(f"| {e} | " + " | ".join(f"{agree.loc[e, o]:.2f}" for o in exps) + " |")
    L += ["", f"## Imagens entre as {pct}% piores", ""]
    L += [f"- **{k}**: {v} imagem(ns)" for k, v in counts.items() if k]
    if not all_hard.empty:
        L += ["", "### Difíceis para todos (provável problema na imagem/anotação)", ""]
        L += [f"- `{n}` — RER médio entre modelos {r['rer_medio_modelos']:.1f}%" for n, r in all_hard.head(15).iterrows()]
    L += ["", "Arquivos: `erro_por_imagem_entre_modelos.csv`, `dificeis_para_todos.png`, `erro_entre_modelos.png`."]
    with open(os.path.join(out_dir, "indicios_comparacao.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
