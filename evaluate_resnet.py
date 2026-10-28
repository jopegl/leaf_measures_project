import os
import sys
import re
import glob
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.datasets import LSIDBeansMeasuresDataset
from common.metrics import calculate_measure_metrics
from common.normalization import build_target_tensor, denormalize_columns, MEASURE_ORDER
from models.resnet_regressor import ResNetRegressor
from configs import resnet_config as cfg

GRAPHICS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graficos')
CSV_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados_csv')
DATASET_ROOT = os.path.join('Leaf on Stem Image Dataset Beans (LSID-Beans)', 'cross-validation')
CHECKPOINT_DIR = os.path.join('runs', 'resnet50')

MODEL_TAG = 'resnet'
METRIC_KEYS = ('mae', 'rmse', 'rer', 'r2')


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def discover_cv_splits():
    pairs = []
    if not os.path.isdir(DATASET_ROOT):
        print(f'[AVISO] Diretório de cross-validation não encontrado: {DATASET_ROOT}')
        return pairs
    for cv_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, 'cv_*'))):
        cv_match = re.search(r'cv_(\d+)$', cv_dir)
        if not cv_match:
            continue
        cv = cv_match.group(1)
        for split_dir in sorted(glob.glob(os.path.join(cv_dir, 'split_*'))):
            split_match = re.search(r'split_(\d+)$', split_dir)
            if not split_match:
                continue
            split = split_match.group(1)
            pairs.append((cv, split))
    pairs.sort(key=lambda p: (int(p[0]), int(p[1])))
    return pairs


def _plot_pred_vs_real(measure_name, preds, targets, save_dir, tag):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.5, s=15, edgecolor='k', linewidth=0.3)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, 'r--', label='y = x (ideal)')
    ax.set_xlabel(f'{measure_name.capitalize()} real')
    ax.set_ylabel(f'{measure_name.capitalize()} predito')
    ax.set_title(f'{tag} — Predito vs Real: {measure_name.capitalize()}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{measure_name}_predito_vs_real.png'), dpi=150)
    plt.close(fig)


def _plot_error_hist(measure_name, values, save_dir, tag, label, filename_suffix, color):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30, color=color, edgecolor='black', alpha=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel('Frequência')
    ax.set_title(f'{tag} — {label}: {measure_name.capitalize()}')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{measure_name}_{filename_suffix}.png'), dpi=150)
    plt.close(fig)


def _plot_rer_boxplot(rer_per_measure, save_dir, tag):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [rer_per_measure[name] for name in MEASURE_ORDER]
    ax.boxplot(data, labels=[n.capitalize() for n in MEASURE_ORDER], showfliers=True)
    ax.set_ylabel('Erro Relativo por amostra — RER (%)')
    ax.set_title(f'{tag} — Distribuição do RER por medida')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'rer_boxplot_todas_medidas.png'), dpi=150)
    plt.close(fig)


def _bar_with_labels(ax, names, values, title, color):
    bars = ax.bar(names, values, color=color, edgecolor='black')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.2f}', ha='center', va='bottom', fontsize=8)


def _plot_metrics_dashboard(metrics_by_measure, save_dir, tag, filename='dashboard_metricas.png'):
    names = [n.capitalize() for n in MEASURE_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _bar_with_labels(axes[0, 0], names, [metrics_by_measure[n]['mae'] for n in MEASURE_ORDER], 'MAE', 'steelblue')
    _bar_with_labels(axes[0, 1], names, [metrics_by_measure[n]['rmse'] for n in MEASURE_ORDER], 'RMSE', 'darkorange')
    _bar_with_labels(axes[1, 0], names, [metrics_by_measure[n]['rer'] for n in MEASURE_ORDER], 'RER (%)', 'crimson')
    _bar_with_labels(axes[1, 1], names, [metrics_by_measure[n]['r2'] for n in MEASURE_ORDER], 'R²', 'seagreen')
    fig.suptitle(f'{tag} — Resumo de métricas por medida', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close(fig)


def _plot_rer_focus(metrics_by_measure, save_dir, tag):
    items = sorted(((n, metrics_by_measure[n]['rer']) for n in MEASURE_ORDER), key=lambda x: x[1], reverse=True)
    names = [i[0].capitalize() for i in items]
    values = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color='crimson', edgecolor='black')
    ax.set_ylabel('RER (%)')
    ax.set_title(f'{tag} — RER por medida (ordenado)')
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'rer_foco_ordenado.png'), dpi=150)
    plt.close(fig)


def generate_metric_plots(tag, results, preds_np, targets_np, save_dir):
    _ensure_dir(save_dir)
    rer_per_measure = {}
    for name in MEASURE_ORDER:
        preds = preds_np[name]
        targets = targets_np[name]
        errors = preds - targets
        abs_errors = np.abs(errors)
        rel_errors = np.abs(errors) / (np.abs(targets) + 1e-08) * 100.0
        rer_per_measure[name] = rel_errors
        _plot_pred_vs_real(name, preds, targets, save_dir, tag)
        _plot_error_hist(name, abs_errors, save_dir, tag, 'Erro Absoluto', 'erro_absoluto_hist', 'steelblue')
        _plot_error_hist(name, errors, save_dir, tag, 'Erro (pred - real)', 'erro_hist', 'darkorange')
        _plot_error_hist(name, rel_errors, save_dir, tag, 'RER por amostra (%)', 'rer_hist', 'crimson')
    _plot_rer_boxplot(rer_per_measure, save_dir, tag)
    _plot_metrics_dashboard(results, save_dir, tag)
    _plot_rer_focus(results, save_dir, tag)
    print(f'Gráficos salvos em: {save_dir}')


def save_predictions_csv(tag, sample_names, preds_np, targets_np, sub_dir=None):
    save_dir = os.path.join(CSV_ROOT, tag)
    if sub_dir:
        save_dir = os.path.join(save_dir, sub_dir)
    _ensure_dir(save_dir)
    for name in MEASURE_ORDER:
        path = os.path.join(save_dir, f'{name}_predictions.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'predicted', 'real'])
            for sample_name, pred, real in zip(sample_names, preds_np[name], targets_np[name]):
                writer.writerow([sample_name, float(pred), float(real)])
        print(f'CSV de predições salvo: {path}')


def plot_cv_average_dashboard(cv, cv_avg_metrics, save_dir_root, tag):
    save_dir_cv = os.path.join(save_dir_root, f'cv{cv}', 'media_ponderada')
    _ensure_dir(save_dir_cv)
    _plot_metrics_dashboard(cv_avg_metrics, save_dir_cv, f'{tag} — Média ponderada CV{cv}')
    _plot_rer_focus(cv_avg_metrics, save_dir_cv, f'{tag} — Média ponderada CV{cv}')
    print(f'Dashboard da média ponderada do CV{cv} salvo em: {save_dir_cv}')


def plot_general_summary(cv_averages, overall_avg, save_dir, tag):
    _ensure_dir(save_dir)

    _plot_metrics_dashboard(overall_avg['metrics'], save_dir, f'{tag} — Média ponderada TOTAL',
                             filename='dashboard_metricas_total.png')
    _plot_rer_focus(overall_avg['metrics'], save_dir, f'{tag} — Média ponderada TOTAL')

    cvs = sorted(cv_averages.keys(), key=lambda c: int(c))
    if not cvs:
        print(f'Gráficos gerais salvos em: {save_dir}')
        return

    x = np.arange(len(MEASURE_ORDER))
    width = 0.8 / max(len(cvs), 1)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, cv in enumerate(cvs):
        vals = [cv_averages[cv]['metrics'][m]['rer'] for m in MEASURE_ORDER]
        ax.bar(x + i * width, vals, width, label=f'CV{cv}', edgecolor='black')
    ax.set_xticks(x + width * (len(cvs) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in MEASURE_ORDER], rotation=20)
    ax.set_ylabel('RER médio ponderado (%)')
    ax.set_title(f'{tag} — RER médio ponderado por CV e medida')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'comparacao_rer_entre_cvs.png'), dpi=150)
    plt.close(fig)
    print(f'Gráficos gerais salvos em: {save_dir}')


def evaluate_resnet(chosen_cv, chosen_split, modo_leve):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    caminho_split = os.path.join(DATASET_ROOT, f'cv_{chosen_cv}', f'split_{chosen_split}', 'test.csv')
    caminho_modelo = os.path.join(CHECKPOINT_DIR, f'resnet50_best_model_cv{chosen_cv}_split{chosen_split}.pth')

    if not os.path.isfile(caminho_split):
        print(f'[AVISO] CV{chosen_cv}/Split{chosen_split}: split de teste não encontrado ({caminho_split}). Ignorando.')
        return None
    if not os.path.isfile(caminho_modelo):
        print(f'[AVISO] CV{chosen_cv}/Split{chosen_split}: checkpoint não encontrado ({caminho_modelo}). '
              f'Provavelmente ainda não treinado — ignorando.')
        return None

    test_dataset = LSIDBeansMeasuresDataset(root_dir=cfg.ROOT, split_file=caminho_split, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )
    model = ResNetRegressor(backbone_name=cfg.BACKBONE, num_outputs=cfg.NUM_OUTPUTS, pretrained=False).to(device)

    try:
        checkpoint = torch.load(caminho_modelo, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as exc:
        print(f'[AVISO] CV{chosen_cv}/Split{chosen_split}: falha ao carregar checkpoint ({exc}). Ignorando.')
        return None
    model.eval()

    iteracao = checkpoint.get('iteration')
    print(f'\n>>> Avaliando ResNet — CV{chosen_cv} / Split{chosen_split}')
    print(f'Checkpoint carregado: {caminho_modelo} (iteração {iteracao})')
    print(f'Imagens de teste: {len(test_dataset)}')

    preds_by_measure = {name: [] for name in MEASURE_ORDER}
    targets_by_measure = {name: [] for name in MEASURE_ORDER}
    sample_names = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)
            out = model(images)
            out_denorm = denormalize_columns(out.detach().cpu())
            target_denorm = denormalize_columns(target.detach().cpu())
            for name in MEASURE_ORDER:
                preds_by_measure[name].append(out_denorm[name])
                targets_by_measure[name].append(target_denorm[name])
            sample_names.extend(f'{leaf}/{img}' for leaf, img in zip(batch['leaf_number'], batch['image_name']))

    results = {}
    preds_np, targets_np = {}, {}
    print('-' * 60)
    print(f'Métricas — CV{chosen_cv} / Split{chosen_split}')
    for name in MEASURE_ORDER:
        preds = torch.cat(preds_by_measure[name])
        targets = torch.cat(targets_by_measure[name])
        mae, rmse, rer, r2 = calculate_measure_metrics(preds, targets)
        results[name] = {'mae': mae, 'rmse': rmse, 'rer': rer, 'r2': r2}
        preds_np[name] = preds.numpy()
        targets_np[name] = targets.numpy()
        print(f'  {name.capitalize():<10} MAE={mae:.4f}  RMSE={rmse:.4f}  RER={rer:.2f}%  R²={r2:.4f}')
    print('-' * 60)

    if not modo_leve:
        sub_dir = os.path.join(f'cv{chosen_cv}', f'split{chosen_split}')
        generate_metric_plots(MODEL_TAG, results, preds_np, targets_np,
                               save_dir=os.path.join(GRAPHICS_ROOT, MODEL_TAG, sub_dir))
        save_predictions_csv(MODEL_TAG, sample_names, preds_np, targets_np, sub_dir=sub_dir)

    return {'cv': chosen_cv, 'split': chosen_split, 'n_samples': len(test_dataset), 'metrics': results}


def compute_cv_weighted_averages(results_list):
    by_cv = {}
    for r in results_list:
        by_cv.setdefault(r['cv'], []).append(r)
    cv_averages = {}
    for cv, entries in by_cv.items():
        total_n = sum(e['n_samples'] for e in entries)
        avg = {}
        for name in MEASURE_ORDER:
            avg[name] = {}
            for key in METRIC_KEYS:
                avg[name][key] = (sum(e['metrics'][name][key] * e['n_samples'] for e in entries) / total_n
                                   if total_n > 0 else float('nan'))
        cv_averages[cv] = {'n_samples': total_n, 'n_splits': len(entries), 'metrics': avg}
    return cv_averages


def compute_overall_weighted_average(results_list):
    total_n = sum(r['n_samples'] for r in results_list)
    avg = {}
    for name in MEASURE_ORDER:
        avg[name] = {}
        for key in METRIC_KEYS:
            avg[name][key] = (sum(r['metrics'][name][key] * r['n_samples'] for r in results_list) / total_n
                               if total_n > 0 else float('nan'))
    return {'n_samples': total_n, 'metrics': avg}


def write_summary_spreadsheet(all_pairs, results_by_pair, cv_averages, overall_avg, path):
    _ensure_dir(os.path.dirname(path))

    cvs_ordenados = sorted({cv for cv, _ in all_pairs}, key=lambda c: int(c))

    columns = []
    for cv in cvs_ordenados:
        splits_do_cv = sorted((s for c, s in all_pairs if c == cv), key=lambda s: int(s))
        for split in splits_do_cv:
            columns.append((f'cv{cv} fold{split}', 'fold', cv, split))
        columns.append((f'cv{cv} weighted av', 'cv_avg', cv, None))
    columns.append(('TOTAL weighted av', 'total_avg', None, None))

    header = ['Resnet50'] + [label for label, _, _, _ in columns]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in MEASURE_ORDER:
            row = [name.capitalize()]
            for _, kind, cv, split in columns:
                if kind == 'fold':
                    r = results_by_pair.get((cv, split))
                    row.append(f"{r['metrics'][name]['rer']:.2f}%" if r is not None else '-')
                elif kind == 'cv_avg':
                    row.append(f"{cv_averages[cv]['metrics'][name]['rer']:.2f}%" if cv in cv_averages else '-')
                else:
                    row.append(f"{overall_avg['metrics'][name]['rer']:.2f}%" if overall_avg['n_samples'] > 0 else '-')
            writer.writerow(row)

    print(f'\nPlanilha de RER salva em: {path}')


def print_summary(cv_averages, overall_avg):
    print('\n' + '#' * 60)
    print('RESUMO GERAL — ResNetRegressor')
    print('#' * 60)
    for cv in sorted(cv_averages.keys(), key=lambda c: int(c)):
        avg = cv_averages[cv]
        print(f"\n--- CV{cv} (média ponderada, {avg['n_splits']} split(s), {avg['n_samples']} amostras) ---")
        for name in MEASURE_ORDER:
            m = avg['metrics'][name]
            print(f"  {name.capitalize():<10} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  RER={m['rer']:.2f}%  R²={m['r2']:.4f}")
    print(f"\n--- MÉDIA PONDERADA TOTAL (todos os CVs, {overall_avg['n_samples']} amostras) ---")
    for name in MEASURE_ORDER:
        m = overall_avg['metrics'][name]
        print(f"  {name.capitalize():<10} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  RER={m['rer']:.2f}%  R²={m['r2']:.4f}")
    print('\n' + '#' * 60)
    print(f'Todos os gráficos foram salvos em: {GRAPHICS_ROOT}')


def perguntar_modo():
    while True:
        resposta = input('Rodar modo completo (com gráficos e CSVs) ou leve (apenas planilha de RER)? [completo/leve]: ').strip().lower()
        if resposta in ('completo', 'c'):
            return False
        if resposta in ('leve', 'l'):
            return True
        print('Resposta inválida, digite "completo" ou "leve".')


def main():
    modo_leve = perguntar_modo()

    all_pairs = discover_cv_splits()
    if not all_pairs:
        print('Nenhuma combinação de CV/split encontrada em:', DATASET_ROOT)
        return

    print(f'Combinações de CV/split encontradas: {len(all_pairs)}')
    for cv, split in all_pairs:
        print(f'  CV{cv} / Split{split}')

    results_by_pair = {}
    results_list = []
    for cv, split in all_pairs:
        r = evaluate_resnet(cv, split, modo_leve)
        results_by_pair[(cv, split)] = r
        if r is not None:
            results_list.append(r)

    if not results_list:
        print('\nNenhum checkpoint de ResNet foi encontrado para nenhuma combinação de CV/split. '
              'Nada a resumir ainda.')
        write_summary_spreadsheet(
            all_pairs, results_by_pair, {}, {'n_samples': 0, 'metrics': {
                name: {key: float('nan') for key in METRIC_KEYS} for name in MEASURE_ORDER
            }},
            os.path.join(CSV_ROOT, MODEL_TAG, 'resumo_metricas.csv'),
        )
        return

    cv_averages = compute_cv_weighted_averages(results_list)
    overall_avg = compute_overall_weighted_average(results_list)

    if not modo_leve:
        for cv, avg in cv_averages.items():
            plot_cv_average_dashboard(cv, avg['metrics'], os.path.join(GRAPHICS_ROOT, MODEL_TAG), MODEL_TAG)
        plot_general_summary(cv_averages, overall_avg, os.path.join(GRAPHICS_ROOT, MODEL_TAG, 'geral'), MODEL_TAG)

    print_summary(cv_averages, overall_avg)

    write_summary_spreadsheet(
        all_pairs, results_by_pair, cv_averages, overall_avg,
        os.path.join(CSV_ROOT, MODEL_TAG, 'resumo_metricas.csv'),
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompido pelo usuário.')
        sys.exit(0)