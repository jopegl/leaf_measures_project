"""Gráficos de avaliação (mesmos do evaluate_resnet.py / evaluate_test.py antigos)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from leaf_measures.data.normalization import MEASURE_ORDER  # noqa: E402

DPI = 150


def _save(fig, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=DPI)
    plt.close(fig)


def plot_pred_vs_real(name, preds, targets, save_dir, tag):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.5, s=15, edgecolor="k", linewidth=0.3)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, "r--", label="y = x (ideal)")
    ax.set_xlabel(f"{name.capitalize()} real")
    ax.set_ylabel(f"{name.capitalize()} predito")
    ax.set_title(f"{tag} — Predito vs Real: {name.capitalize()}")
    ax.legend()
    _save(fig, save_dir, f"{name}_predito_vs_real.png")


def plot_hist(name, values, save_dir, tag, label, suffix, color):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30, color=color, edgecolor="black", alpha=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel("Frequência")
    ax.set_title(f"{tag} — {label}: {name.capitalize()}")
    _save(fig, save_dir, f"{name}_{suffix}.png")


def plot_rer_boxplot(rer_per_measure, save_dir, tag):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([rer_per_measure[n] for n in MEASURE_ORDER], showfliers=True)
    ax.set_xticks(range(1, len(MEASURE_ORDER) + 1), [n.capitalize() for n in MEASURE_ORDER])
    ax.set_ylabel("Erro Relativo por amostra — RER (%)")
    ax.set_title(f"{tag} — Distribuição do RER por medida")
    _save(fig, save_dir, "rer_boxplot_todas_medidas.png")


def _bars(ax, names, values, title, color, fmt="{:.2f}"):
    bars = ax.bar(names, values, color=color, edgecolor="black")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8)


def plot_metrics_dashboard(metrics, save_dir, tag, filename="dashboard_metricas.png"):
    names = [n.capitalize() for n in MEASURE_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title, color) in zip(
        axes.flat,
        [("mae", "MAE", "steelblue"), ("rmse", "RMSE", "darkorange"),
         ("rer", "RER (%)", "crimson"), ("r2", "R²", "seagreen")],
    ):
        _bars(ax, names, [metrics[n][key] for n in MEASURE_ORDER], title, color)
    fig.suptitle(f"{tag} — Resumo de métricas por medida", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, filename), dpi=DPI)
    plt.close(fig)


def plot_rer_focus(metrics, save_dir, tag):
    items = sorted(((n, metrics[n]["rer"]) for n in MEASURE_ORDER), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    _bars(ax, [i[0].capitalize() for i in items], [i[1] for i in items], f"{tag} — RER por medida (ordenado)",
          "crimson", fmt="{:.2f}%")
    ax.set_ylabel("RER (%)")
    _save(fig, save_dir, "rer_foco_ordenado.png")


def fold_plots(tag, metrics, preds_np, targets_np, save_dir):
    rer_per_measure = {}
    for n in MEASURE_ORDER:
        p, t = preds_np[n], targets_np[n]
        err = p - t
        rel = np.abs(err) / (np.abs(t) + 1e-8) * 100.0
        rer_per_measure[n] = rel
        plot_pred_vs_real(n, p, t, save_dir, tag)
        plot_hist(n, np.abs(err), save_dir, tag, "Erro Absoluto", "erro_absoluto_hist", "steelblue")
        plot_hist(n, err, save_dir, tag, "Erro (pred - real)", "erro_hist", "darkorange")
        plot_hist(n, rel, save_dir, tag, "RER por amostra (%)", "rer_hist", "crimson")
    plot_rer_boxplot(rer_per_measure, save_dir, tag)
    plot_metrics_dashboard(metrics, save_dir, tag)
    plot_rer_focus(metrics, save_dir, tag)


def summary_plots(tag, cv_averages: dict, overall: dict, save_dir_root):
    for cv, avg in cv_averages.items():
        d = os.path.join(save_dir_root, f"cv{cv}", "media_ponderada")
        plot_metrics_dashboard(avg["metrics"], d, f"{tag} — Média ponderada CV{cv}")
        plot_rer_focus(avg["metrics"], d, f"{tag} — Média ponderada CV{cv}")

    geral = os.path.join(save_dir_root, "geral")
    plot_metrics_dashboard(overall["metrics"], geral, f"{tag} — Média ponderada TOTAL", "dashboard_metricas_total.png")
    plot_rer_focus(overall["metrics"], geral, f"{tag} — Média ponderada TOTAL")

    cvs = sorted(cv_averages, key=int)
    if not cvs:
        return
    x = np.arange(len(MEASURE_ORDER))
    width = 0.8 / len(cvs)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, cv in enumerate(cvs):
        ax.bar(x + i * width, [cv_averages[cv]["metrics"][m]["rer"] for m in MEASURE_ORDER], width,
               label=f"CV{cv}", edgecolor="black")
    ax.set_xticks(x + width * (len(cvs) - 1) / 2, [m.capitalize() for m in MEASURE_ORDER], rotation=20)
    ax.set_ylabel("RER médio ponderado (%)")
    ax.set_title(f"{tag} — RER médio ponderado por CV e medida")
    ax.legend()
    _save(fig, geral, "comparacao_rer_entre_cvs.png")


def plot_model_comparison(summaries: dict, save_path):
    """summaries: {rotulo: {medida: {mae, rmse, rer, r2}}} (média ponderada total de cada experimento)."""
    labels = list(summaries)
    x = np.arange(len(MEASURE_ORDER))
    width = 0.8 / len(labels)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title) in zip(axes.flat, [("mae", "MAE"), ("rmse", "RMSE"), ("rer", "RER (%)"), ("r2", "R²")]):
        for i, lab in enumerate(labels):
            off = (i - (len(labels) - 1) / 2) * width
            ax.bar(x + off, [summaries[lab][m][key] for m in MEASURE_ORDER], width, label=lab,
                   color=cmap(i % 10), edgecolor="black")
        ax.set_xticks(x, [m.capitalize() for m in MEASURE_ORDER], rotation=25)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Comparação entre experimentos: " + " vs ".join(labels), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


def save_input_examples(images, inputs, save_dir, title="Entrada da rede"):
    """images: [N,3,H,W] em [0,1]; inputs: tensor que entra na rede (mostra os 3 primeiros canais)."""
    os.makedirs(save_dir, exist_ok=True)
    for i in range(images.shape[0]):
        orig = images[i].clamp(0, 1).permute(1, 2, 0).numpy()
        inp = inputs[i, :3]
        inp = (inp - inp.min()) / (inp.max() - inp.min() + 1e-8)  # reescala p/ visualizar (pode estar normalizada)
        n_extra = inputs.shape[1] - 3
        fig, axes = plt.subplots(1, 2 + n_extra, figsize=(5 * (2 + n_extra), 5))
        axes[0].imshow(orig)
        axes[0].set_title("Imagem original")
        axes[1].imshow(inp.permute(1, 2, 0).numpy())
        axes[1].set_title(f"{title} (RGB)")
        for c in range(n_extra):
            axes[2 + c].imshow(inputs[i, 3 + c].numpy(), cmap="gray")
            axes[2 + c].set_title(f"Canal extra {c + 1}")
        for ax in axes:
            ax.axis("off")
        _save(fig, save_dir, f"exemplo_{i + 1:02d}.png")
