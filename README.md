# leaf_measures_project

Regressão das medidas de folhas de feijão (comprimento, largura, perímetro e área) no
dataset **LSID-Beans**, com backbones do torchvision e cross-validation (5 CVs × 5 splits).

> O código da DeepLab foi removido nesta reorganização. Ele está preservado na tag
> `deeplab-final` (`git checkout deeplab-final`).

## Estrutura

```
train.py / evaluate.py / compare.py     pontos de entrada
configs/
  base.yaml                             todos os parâmetros com valores padrão
  experiments/*.yaml                    um arquivo por experimento (herda da base)
leaf_measures/
  config.py                             YAML com herança (_base_) + overrides --set
  registry.py                           registro por nome de modelos e modos de entrada
  data/        annotation, splits, normalization, datasets, folds
  models/      backbones (fábrica torchvision), regressor, inputs (rgb/hadamard/mask_concat)
  training/    trainer (loop por fold), losses, checkpoint, run_log
  evaluation/  inference, metrics, plots, evaluator
tools/migrate_old_runs.py               copia checkpoints do layout antigo
```

Saídas:

```
runs/<experimento>/config.yaml               snapshot da config usada
runs/<experimento>/training_log.csv          todas as validações de todos os folds
runs/<experimento>/<fold>/best.pth           melhor val loss (usado no teste)
runs/<experimento>/<fold>/last.pth           última validação (usado para retomar)
results/<experimento>/                        métricas, gráficos, CSVs e summary.json do teste
```

## Uso

```bash
pip install -r requirements.txt

# teste rápido do pipeline (1 fold, 60 iterações)
python train.py    -c configs/experiments/smoke_test.yaml
python evaluate.py -c configs/experiments/smoke_test.yaml --light

# experimento completo (25 folds)
python train.py    -c configs/experiments/resnet50_hadamard.yaml
python evaluate.py -c configs/experiments/resnet50_hadamard.yaml

# só alguns folds
python train.py -c configs/experiments/resnet50_rgb.yaml --folds cv1_split1 cv1_split2

# mudar parâmetros sem criar arquivo
python train.py -c configs/experiments/resnet50_rgb.yaml \
    --set model.backbone=resnet101 experiment.name=resnet101_rgb train.batch_size=2

# comparar experimentos já avaliados
python compare.py resnet50_rgb resnet50_hadamard efficientnet_b0_rgb
```

O treino retoma sozinho de onde parou (`last.pth`). Para recomeçar um fold do zero, apague a
pasta `runs/<experimento>/<fold>/`.

> **Use um `experiment.name` novo sempre que mudar algo que altere o modelo.** O nome
> define a pasta em `runs/`. Se você trocar o backbone ou o modo de entrada via `--set` e
> mantiver o nome, o treino para com um erro claro em vez de misturar checkpoints de
> arquiteturas diferentes.

## Como adicionar coisas novas

### Um novo backbone
Basta criar um YAML. Qualquer nome de `torchvision.models.get_model` funciona
(`resnet18/34/50/101`, `resnext50_32x4d`, `wide_resnet50_2`, `efficientnet_b0..b7`,
`efficientnet_v2_s`, `convnext_tiny`, `regnet_y_8gf`, `densenet121`, `swin_t`...):

```yaml
# configs/experiments/resnext50_hadamard.yaml
_base_: resnet50_hadamard.yaml
experiment: {label: ResNeXt-50 + Hadamard}
model: {backbone: resnext50_32x4d}
```

Os ViT do torchvision exigem entrada 224×224, e as imagens aqui são 512×512.

### Um novo modo de entrada
Em `leaf_measures/models/inputs.py`:

```python
@INPUT_MODES.register("so_folha", in_channels=3, needs_masks=True)
def so_folha(image, masks):
    return image * masks[:, :1]        # só a folha, sem o marcador
```

Depois, no YAML: `input: {mode: so_folha}`. Se `in_channels != 3`, a primeira conv do
backbone é adaptada automaticamente: mantém os pesos RGB pré-treinados e inicializa os canais
extras conforme `model.extra_channels_init`. O padrão é `zeros`, como no ResNetMaskRegressor
original, em que a rede começa idêntica à pré-treinada em RGB; a alternativa é `mean`, a média
dos pesos RGB. O dataset só carrega as máscaras quando `needs_masks=True`.

### Uma nova arquitetura (outro head, dois ramos, multi-tarefa...)
Crie um arquivo em `leaf_measures/models/`, registre o modelo e importe o arquivo em
`leaf_measures/models/__init__.py`:

```python
@MODELS.register("dual_branch")
class DualBranch(nn.Module):
    @classmethod
    def from_config(cls, model_cfg, in_channels, num_outputs): ...
    def forward(self, x): ...                 # -> [B, 4] medidas normalizadas
    def backbone_params(self): ...
    def head_params(self): ...
    def set_backbone_trainable(self, flag): ...
```

No YAML: `model: {name: dual_branch, ...}`. Chaves extras em `model:` chegam em `model_cfg`.

## Migrando do layout antigo

| antes                          | agora                                                      |
|--------------------------------|------------------------------------------------------------|
| `train_resnet_measures.py`     | `train.py -c configs/experiments/resnet50_rgb.yaml`        |
| `train_resnet_hadamard.py`     | `train.py -c configs/experiments/resnet50_hadamard.yaml`   |
| `train_resnet_mask.py`         | `train.py -c configs/experiments/resnet50_mask.yaml`       |
| `evaluate_resnet.py`           | `evaluate.py -c ...` (todos os folds, médias ponderadas)   |
| `evaluate_test.py` (menu)      | `evaluate.py` por experimento + `compare.py`               |
| `configs/resnet_config.py`     | `configs/base.yaml`                                        |
| `common/*`                     | `leaf_measures/data`, `training`, `evaluation`             |
| `graficos/`, `resultados_csv/` | `results/<experimento>/`                                   |

Os checkpoints antigos de `resnet50`, `resnet50_hadamard` e `resnet50_mask` carregam direto,
porque os nomes e formatos das camadas foram mantidos. O ResNetMaskRegressor equivale ao modo
`mask_concat` com `extra_channels_init: zeros`. Para copiar os checkpoints para o layout novo:

```bash
python tools/migrate_old_runs.py          # mostra o que vai copiar
python tools/migrate_old_runs.py --apply
python evaluate.py -c configs/experiments/resnet50_rgb.yaml
```


### Diferenças de comportamento em relação aos scripts antigos
- O treino retoma do `last.pth` (última validação), não do melhor checkpoint. Assim não se
  perde progresso nem o estado do scheduler.
- O descongelamento do backbone é decidido pela iteração
  (`freeze_backbone_validations × val_interval`). Isso dá o mesmo resultado de antes e
  continua correto ao retomar.
- O seed é fixo por fold (`experiment.seed + 100·cv + split`).
- `data.imagenet_normalize` vale `false` na base, como antes (a entrada fica em [0, 1]).
  Para backbones novos, vale testar com `true`. No modo `hadamard`, a máscara é aplicada
  depois da normalização.
- A média ponderada (por nº de amostras de teste) por CV e a total agora valem para qualquer
  experimento, não só para a ResNet.
