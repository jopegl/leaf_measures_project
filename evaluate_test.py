import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from common.datasets import LSIDBeansMeasuresDataset, LSIDBeansMaskedMeasuresDataset, LSIDBeansDataset
from common.metrics import calculate_measure_metrics, calculate_miou
from common.normalization import build_target_tensor, denormalize_columns, MEASURE_ORDER

GRAPHICS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graficos')
CSV_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados_csv')
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

MODEL_INFO = {
    'resnet': {'nome': 'ResNetRegressor', 'rotulo': 'ResNet', 'cor': 'steelblue'},
    'resnet_mask': {'nome': 'ResNetMaskRegressor', 'rotulo': 'ResNet + máscara', 'cor': 'seagreen'},
    'deeplab': {'nome': 'DeepLab multi-tarefa', 'rotulo': 'DeepLab', 'cor': 'darkorange'},
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _plot_pred_vs_real(measure_name, preds, targets, save_dir, model_tag):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.5, s=15, edgecolor='k', linewidth=0.3)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, 'r--', label='y = x (ideal)')
    ax.set_xlabel(f'{measure_name.capitalize()} real')
    ax.set_ylabel(f'{measure_name.capitalize()} predito')
    ax.set_title(f'{model_tag} — Predito vs Real: {measure_name.capitalize()}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{measure_name}_predito_vs_real.png'), dpi=150)
    plt.close(fig)


def _plot_error_hist(measure_name, values, save_dir, model_tag, label, filename_suffix, color):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30, color=color, edgecolor='black', alpha=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel('Frequência')
    ax.set_title(f'{model_tag} — {label}: {measure_name.capitalize()}')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{measure_name}_{filename_suffix}.png'), dpi=150)
    plt.close(fig)


def _plot_rer_boxplot(rer_per_measure, save_dir, model_tag):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [rer_per_measure[name] for name in MEASURE_ORDER]
    ax.boxplot(data, labels=[n.capitalize() for n in MEASURE_ORDER], showfliers=True)
    ax.set_ylabel('Erro Relativo por amostra — RER (%)')
    ax.set_title(f'{model_tag} — Distribuição do RER por medida')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'rer_boxplot_todas_medidas.png'), dpi=150)
    plt.close(fig)


def _bar_with_labels(ax, names, values, title, color):
    bars = ax.bar(names, values, color=color, edgecolor='black')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.2f}', ha='center', va='bottom', fontsize=8)


def _plot_metrics_dashboard(results, save_dir, model_tag):
    names = [n.capitalize() for n in MEASURE_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _bar_with_labels(axes[0, 0], names, [results[n]['mae'] for n in MEASURE_ORDER], 'MAE', 'steelblue')
    _bar_with_labels(axes[0, 1], names, [results[n]['rmse'] for n in MEASURE_ORDER], 'RMSE', 'darkorange')
    _bar_with_labels(axes[1, 0], names, [results[n]['rer'] for n in MEASURE_ORDER], 'RER (%)', 'crimson')
    _bar_with_labels(axes[1, 1], names, [results[n]['r2'] for n in MEASURE_ORDER], 'R²', 'seagreen')
    fig.suptitle(f'{model_tag} — Resumo de métricas por medida', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, 'dashboard_metricas.png'), dpi=150)
    plt.close(fig)


def _plot_rer_focus(results, save_dir, model_tag):
    items = sorted(((n, results[n]['rer']) for n in MEASURE_ORDER), key=lambda x: x[1], reverse=True)
    names = [i[0].capitalize() for i in items]
    values = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color='crimson', edgecolor='black')
    ax.set_ylabel('RER (%)')
    ax.set_title(f'{model_tag} — RER por medida (ordenado)')
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'rer_foco_ordenado.png'), dpi=150)
    plt.close(fig)


def generate_metric_plots(model_tag, results, preds_np, targets_np):
    save_dir = os.path.join(GRAPHICS_ROOT, model_tag)
    _ensure_dir(save_dir)
    rer_per_measure = {}
    for name in MEASURE_ORDER:
        preds = preds_np[name]
        targets = targets_np[name]
        errors = preds - targets
        abs_errors = np.abs(errors)
        rel_errors = np.abs(errors) / (np.abs(targets) + 1e-08) * 100.0
        rer_per_measure[name] = rel_errors
        _plot_pred_vs_real(name, preds, targets, save_dir, model_tag)
        _plot_error_hist(name, abs_errors, save_dir, model_tag, 'Erro Absoluto', 'erro_absoluto_hist', 'steelblue')
        _plot_error_hist(name, errors, save_dir, model_tag, 'Erro (pred - real)', 'erro_hist', 'darkorange')
        _plot_error_hist(name, rel_errors, save_dir, model_tag, 'RER por amostra (%)', 'rer_hist', 'crimson')
    _plot_rer_boxplot(rer_per_measure, save_dir, model_tag)
    _plot_metrics_dashboard(results, save_dir, model_tag)
    _plot_rer_focus(results, save_dir, model_tag)
    print(f'Gráficos de métricas salvos em: {save_dir}')


def save_predictions_csv(model_tag, sample_names, preds_np, targets_np, sub_dir=None):
    save_dir = os.path.join(CSV_ROOT, model_tag)
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


def plot_model_comparison(all_results, save_dir):
    by_model = {}
    for r in all_results:
        by_model[r['model']] = r
    present = [k for k in MODEL_INFO if k in by_model]
    if len(present) < 2:
        return
    _ensure_dir(save_dir)
    metrics_keys = ['mae', 'rmse', 'rer', 'r2']
    titles = {'mae': 'MAE', 'rmse': 'RMSE', 'rer': 'RER (%)', 'r2': 'R²'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(MEASURE_ORDER))
    width = 0.8 / len(present)
    for ax, key in zip(axes.flat, metrics_keys):
        for i, model_key in enumerate(present):
            vals = [by_model[model_key]['metrics'][m][key] for m in MEASURE_ORDER]
            offset = (i - (len(present) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=MODEL_INFO[model_key]['rotulo'],
                   color=MODEL_INFO[model_key]['cor'], edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in MEASURE_ORDER], rotation=25)
        ax.set_title(titles[key])
        ax.legend()
    fig.suptitle('Comparação entre modelos: ' + ' vs '.join(MODEL_INFO[k]['rotulo'] for k in present), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, 'comparacao_modelos.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Gráfico de comparação salvo em: {path}')


def _denormalize_image(img_tensor):
    mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
    std = torch.tensor(IMG_STD).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def save_segmentation_examples(model, test_dataset, device, num_classes, model_tag, num_examples=6):
    save_dir = os.path.join(GRAPHICS_ROOT, model_tag, 'segmentacao')
    _ensure_dir(save_dir)
    num_examples = min(num_examples, len(test_dataset))
    vis_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_examples, shuffle=True, num_workers=0)
    cmap = plt.get_cmap('tab20', num_classes)
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in vis_loader:
            images = batch['image'].to(device, non_blocking=True)
            seg_mask = batch['seg_mask']
            seg_out, _ = model(images)
            pred_mask = torch.argmax(seg_out, dim=1).cpu()
            for i in range(images.size(0)):
                if count >= num_examples:
                    break
                img = _denormalize_image(images[i].cpu())
                gt = seg_mask[i].numpy()
                pred = pred_mask[i].numpy()
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img)
                axes[0].set_title('Imagem original')
                axes[0].axis('off')
                axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=num_classes - 1)
                axes[1].set_title('Segmentação real')
                axes[1].axis('off')
                axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1)
                axes[2].set_title('Segmentação predita')
                axes[2].axis('off')
                axes[3].imshow(img)
                axes[3].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1, alpha=0.5)
                axes[3].set_title('Predita sobre a imagem original')
                axes[3].axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, f'exemplo_{count + 1:02d}.png'), dpi=150)
                plt.close(fig)
                count += 1
            break
    print(f'Exemplos de segmentação salvos em: {save_dir}')


def _run_regression_eval(model, test_loader, device, use_masks):
    preds_by_measure = {name: [] for name in MEASURE_ORDER}
    targets_by_measure = {name: [] for name in MEASURE_ORDER}
    sample_names = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)
            if use_masks:
                masks = batch['masks'].to(device, non_blocking=True)
                out = model(images, masks)
            else:
                out = model(images)
            out_denorm = denormalize_columns(out.detach().cpu())
            target_denorm = denormalize_columns(target.detach().cpu())
            for name in MEASURE_ORDER:
                preds_by_measure[name].append(out_denorm[name])
                targets_by_measure[name].append(target_denorm[name])
            sample_names.extend((f'{leaf}/{img}' for leaf, img in zip(batch['leaf_number'], batch['image_name'])))
    return preds_by_measure, targets_by_measure, sample_names


def _report_regression(model_tag, titulo, preds_by_measure, targets_by_measure, sample_names, sub_dir=None):
    results = {}
    preds_np, targets_np = ({}, {})
    print('\n' + '=' * 60)
    print(f'MÉTRICAS FINAIS NO TESTE ({titulo})')
    for name in MEASURE_ORDER:
        preds = torch.cat(preds_by_measure[name])
        targets = torch.cat(targets_by_measure[name])
        mae, rmse, rer, r2 = calculate_measure_metrics(preds, targets)
        results[name] = {'mae': mae, 'rmse': rmse, 'rer': rer, 'r2': r2}
        preds_np[name] = preds.numpy()
        targets_np[name] = targets.numpy()
        print(f'  {name.capitalize():<10} MAE={mae:.4f}  RMSE={rmse:.4f}  RER={rer:.2f}%  R²={r2:.4f}')
    print('=' * 60)
    generate_metric_plots(model_tag, results, preds_np, targets_np)
    save_predictions_csv(model_tag, sample_names, preds_np, targets_np, sub_dir=sub_dir)
    return results


def evaluate_resnet(chosen_cv, chosen_split):
    from models.resnet_regressor import ResNetRegressor
    from configs import resnet_config as cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    caminho_split = os.path.join('Leaf on Stem Image Dataset Beans (LSID-Beans)', 'cross-validation', f'cv_{chosen_cv}', f'split_{chosen_split}', 'test.csv')
    caminho_modelo = os.path.join('runs', 'resnet50', f'resnet50_best_model_cv{chosen_cv}_split{chosen_split}.pth')
    test_dataset = LSIDBeansMeasuresDataset(root_dir=cfg.ROOT, split_file=caminho_split, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    model = ResNetRegressor(backbone_name=cfg.BACKBONE, num_outputs=cfg.NUM_OUTPUTS, pretrained=False).to(device)
    checkpoint = torch.load(caminho_modelo, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Checkpoint carregado: {caminho_modelo} (iteração {checkpoint.get('iteration')})")
    print(f'Imagens de teste: {len(test_dataset)}')
    preds_by_measure, targets_by_measure, sample_names = _run_regression_eval(model, test_loader, device, use_masks=False)
    results = _report_regression('resnet', 'ResNetRegressor', preds_by_measure, targets_by_measure, sample_names,
                                 sub_dir=f'cv{chosen_cv}_split{chosen_split}')
    return {'model': 'resnet', 'metrics': results}


def evaluate_resnet_mask(chosen_cv, chosen_split):
    from models.resnet_mask_regressor import ResNetMaskRegressor
    from configs import resnet_config as cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    caminho_split = os.path.join('Leaf on Stem Image Dataset Beans (LSID-Beans)', 'cross-validation', f'cv_{chosen_cv}', f'split_{chosen_split}', 'test.csv')
    caminho_modelo = os.path.join('runs', 'resnet50_mask', f'resnet50_mask_best_model_cv{chosen_cv}_split{chosen_split}.pth')
    test_dataset = LSIDBeansMaskedMeasuresDataset(root_dir=cfg.ROOT, split_file=caminho_split, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    model = ResNetMaskRegressor(backbone_name=cfg.BACKBONE, num_outputs=cfg.NUM_OUTPUTS, pretrained=False, mask_channels=2).to(device)
    checkpoint = torch.load(caminho_modelo, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Checkpoint carregado: {caminho_modelo} (iteração {checkpoint.get('iteration')})")
    print(f'Imagens de teste: {len(test_dataset)}')
    preds_by_measure, targets_by_measure, sample_names = _run_regression_eval(model, test_loader, device, use_masks=True)
    results = _report_regression('resnet_mask', 'ResNetMaskRegressor, máscaras ground truth', preds_by_measure, targets_by_measure, sample_names,
                                 sub_dir=f'cv{chosen_cv}_split{chosen_split}')
    return {'model': 'resnet_mask', 'metrics': results}


def evaluate_deeplab():
    from models.deeplab_just_measures import DeepLab
    from configs import deeplab_config as cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = LSIDBeansDataset(root_dir=cfg.ROOT, split_file=cfg.TEST_CSV, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    model = DeepLab(num_classes=cfg.NUM_CLASSES, backbone=cfg.BACKBONE, output_stride=cfg.OUTPUT_STRIDE, sync_bn=cfg.SYNC_BN, freeze_bn=cfg.FREEZE_BN).to(device)
    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Checkpoint carregado: {cfg.CHECKPOINT_PATH} (iteração {checkpoint.get('iteration')})")
    print(f'Imagens de teste: {len(test_dataset)}')
    preds_by_measure = {name: [] for name in MEASURE_ORDER}
    targets_by_measure = {name: [] for name in MEASURE_ORDER}
    sample_names = []
    miou_total = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device, non_blocking=True)
            seg_mask = batch['seg_mask'].to(device, non_blocking=True)
            target = build_target_tensor(batch, device)
            seg_out, measures_out = model(images)
            miou_total += calculate_miou(seg_out, seg_mask, cfg.NUM_CLASSES)
            out_denorm = denormalize_columns(measures_out.detach().cpu())
            target_denorm = denormalize_columns(target.detach().cpu())
            for name in MEASURE_ORDER:
                preds_by_measure[name].append(out_denorm[name])
                targets_by_measure[name].append(target_denorm[name])
            sample_names.extend((f'{leaf}/{img}' for leaf, img in zip(batch['leaf_number'], batch['image_name'])))
    miou = miou_total / len(test_loader)
    print(f'\n  mIoU: {miou:.4f}')
    results = _report_regression('deeplab', 'DeepLab multi-tarefa', preds_by_measure, targets_by_measure, sample_names)
    save_segmentation_examples(model, test_dataset, device, cfg.NUM_CLASSES, 'deeplab', num_examples=6)
    return {'model': 'deeplab', 'miou': miou, 'metrics': results}


def print_summary(all_results):
    if not all_results:
        return
    print('\n' + '#' * 60)
    print('RESUMO DA SESSÃO (resultados separados por modelo)')
    print('#' * 60)
    for entry in all_results:
        nome = MODEL_INFO[entry['model']]['nome']
        print(f'\n--- {nome} ---')
        if 'miou' in entry:
            print(f"  mIoU: {entry['miou']:.4f}")
        for name in MEASURE_ORDER:
            m = entry['metrics'][name]
            print(f"  {name.capitalize():<10} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  RER={m['rer']:.2f}%  R²={m['r2']:.4f}")
    print('\n' + '#' * 60)
    print(f'Todos os gráficos foram salvos em: {GRAPHICS_ROOT}')


def _ask_cv_and_fold():
    escolha_cv = input('Digite o CV: ').strip()
    escolha_fold = input('Digite o fold: ').strip()
    return escolha_cv, escolha_fold


def main():
    all_results = []
    while True:
        print('\nQual modelo você deseja avaliar?')
        print('  [1] ResNet')
        print('  [2] DeepLab')
        print('  [3] ResNet com máscara')
        print('  [4] Sair')
        escolha = input('Digite sua opção (1/2/3/4): ').strip()
        if escolha in ('1', 'resnet', 'ResNet'):
            cv, fold = _ask_cv_and_fold()
            all_results.append(evaluate_resnet(cv, fold))
        elif escolha in ('2', 'deeplab', 'DeepLab'):
            all_results.append(evaluate_deeplab())
        elif escolha in ('3', 'mask', 'resnet_mask'):
            cv, fold = _ask_cv_and_fold()
            all_results.append(evaluate_resnet_mask(cv, fold))
        elif escolha in ('4', 'sair', 'exit', 'q'):
            print('Encerrando o script.')
            break
        else:
            print('Opção inválida. Tente novamente.')
            continue
    print_summary(all_results)
    plot_model_comparison(all_results, os.path.join(GRAPHICS_ROOT, 'comparacao'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompido pelo usuário.')
        sys.exit(0)