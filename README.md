# leaf_measures_project

Regressão das medidas de folhas de feijão (comprimento, largura, perímetro e área) no
dataset **LSID-Beans**, com backbones do torchvision e cross-validation (5 CVs × 5 splits).

> O código da DeepLab foi removido nesta reorganização. Ele está preservado na tag
> `deeplab-final` (`git checkout deeplab-final`).

## Estrutura

```
train.py / evaluate.py / compare.py     pontos de entrada
analyze.py                              análise de erros por imagem ou folha (piores/melhores + indícios)
aggregate.py                            predição por folha e ensemble de redes (RER recalculado)
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
  analysis/    features, ranking, patterns, report, aggregation (análise de erros e agregações)
  deep_viewer.py  planilhas de treino e teste no formato do Deep Viewer
tools/migrate_old_runs.py               copia checkpoints do layout antigo
```

Saídas:

```
runs/<experimento>/config.yaml               snapshot da config usada
runs/<experimento>/training_log.csv          todas as validações de todos os folds
runs/<experimento>/<fold>/best.pth           melhor val loss (usado no teste)
runs/<experimento>/<fold>/last.pth           última validação (usado para retomar)
runs/<experimento>/<fold>/deepviewer_train.csv   planilha de treino do Deep Viewer
results/<experimento>/<fold>/deepviewer_test.csv planilha de teste do Deep Viewer
results/<experimento>/                        métricas, gráficos, CSVs e summary.json do teste
results/analise/<experimento>/                indicios.md, piores/melhores, gráficos da análise
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
python compare.py resnet50_rgb resnet50_hadamard efficientnet_b3_rgb
```

O treino retoma sozinho de onde parou (`last.pth`). Para recomeçar um fold do zero, apague a
pasta `runs/<experimento>/<fold>/`.

> **Use um `experiment.name` novo sempre que mudar algo que altere o modelo.** O nome
> define a pasta em `runs/`. Se você trocar o backbone ou o modo de entrada via `--set` e
> mantiver o nome, o treino para com um erro claro em vez de misturar checkpoints de
> arquiteturas diferentes.

## Predição por folha e ensemble de redes

Novas formas de predição, com MAE, RMSE, RER e R² **recalculados** sobre a nova predição. Elas
partem das predições salvas pelo `evaluate.py` (rode-o antes para cada experimento):

```bash
# por FOLHA: a predição da folha é a média (ou mediana) das predições das imagens dela
python aggregate.py resnet50_rgb --leaf mean
python aggregate.py resnet50_rgb --leaf median

# ENSEMBLE: a predição de cada imagem é a média das predições de várias redes
python aggregate.py resnet50_rgb efficientnet_b3_rgb resnet50_hadamard --ensemble mean

# os dois: ensemble por imagem e, depois, agregação por folha
python aggregate.py resnet50_rgb efficientnet_b3_rgb --ensemble mean --leaf mean
```

- **Folha:** a folha é a pasta da imagem (`leaf_xx/`). Em cada fold, junta as imagens dela que
  estão no teste. O valor real da folha é a média dos valores reais das imagens. Se esses valores
  variarem dentro de uma pasta (a mesma folha fotografada em momentos diferentes, por exemplo), o
  script avisa, e `--leaf-key pasta+medidas` trata cada conjunto de medidas anotadas como uma folha
  separada.
- **Ensemble:** usa só os folds avaliados em todos os experimentos e confere que os valores reais
  são idênticos entre eles.
- As médias por CV e total continuam ponderadas pelo nº de amostras de cada fold: imagens no
  ensemble, **folhas** na agregação por folha.

O terminal mostra uma tabela com o RER de cada rede e o da nova predição, calculados sobre as
mesmas imagens. O resultado é gravado como um experimento em `results/<nome>/` (por exemplo,
`results/resnet50_rgb__folha_media/`), no mesmo formato do `evaluate.py`. Por isso ele funciona
direto com as outras ferramentas:

```bash
python compare.py resnet50_rgb resnet50_rgb__folha_media
python analyze.py resnet50_rgb__folha_media     # piores e melhores FOLHAS + indícios
```

## Deep Viewer

O treino e a avaliação gravam as planilhas no formato do Deep Viewer. Cada fold é um experimento
com unidade de passo **iteração**:

- **Treino** (`runs/<exp>/<fold>/deepviewer_train.csv`): uma linha a cada `log_every` iterações,
  com `iteration`, `train_loss`, `val_loss`, `learning_rate` e, para cada medida,
  `train_loss_<medida>`/`val_loss_<medida>`, `train_mae_<medida>`/`val_mae_<medida>`,
  `train_rer_<medida>`/`val_rer_<medida>`, `val_rmse_<medida>` e `val_r2_<medida>`. As colunas de
  treino são médias desde a linha anterior. As de validação só são preenchidas nas iterações com
  validação. Ao retomar um treino, as linhas posteriores ao último checkpoint são descartadas, para
  nenhuma iteração aparecer duas vezes.
- **Teste** (`results/<exp>/<fold>/deepviewer_test.csv`): uma linha por imagem, no formato
  multi-saída: `sample_id`, `true_length`, `pred_length`, …, `true_area`, `pred_area`. Na tela de
  confirmação do Deep Viewer, as quatro saídas devem ficar como **regressão**. Os experimentos
  gerados pelo `aggregate.py` também têm essa planilha (por folha, `sample_id` é a folha).

## Análise de erros: piores, melhores e padrões

Depois de avaliar (`evaluate.py`, pode ser com `--light`):

```bash
python analyze.py resnet50_rgb                        # um modelo
python analyze.py resnet50_rgb efficientnet_b3_rgb    # + comparação entre modelos
python analyze.py resnet50_rgb --folds cv1_split1 cv2_split2 --top-n 16
python analyze.py resnet50_rgb__folha_media           # por FOLHA (gerado pelo aggregate.py)
```

A análise não roda modelo nenhum: ela lê as predições salvas e, para cada imagem, junta o erro
de todos os folds em que ela foi testada. Depois cruza esse erro com características calculadas
da anotação, da imagem e da máscara: tamanho real da folha, fração da foto ocupada pela folha,
escala (via marcador), folha ou marcador encostando na borda, brilho, contraste, nitidez,
saturação, pasta da folha e a divergência entre a área calculada pela máscara + marcador e a
área anotada (que ajuda a achar anotações suspeitas).

Saída em `results/analise/<experimento>/`:

- **`indicios.md`**: os padrões encontrados, em português, do mais forte para o mais fraco.
  Por exemplo: *"Erro maior em imagens com menor fração da imagem ocupada pela folha (todas as
  medidas; ρ de −0,74 a −0,70). Nas 10% piores, mediana 3,3% contra 15,7% nas demais."* Quando
  duas características andam juntas, o relatório avisa que podem ser o mesmo efeito.
- **`imagens/piores_*.png` e `melhores_*.png`**: grades com as N piores e melhores imagens por
  medida, com o contorno da máscara (folha em amarelo, marcador em ciano) e real × predito × RER.
- **`piores_em_varias_medidas.csv`**: imagens ruins em 3 ou mais medidas, em geral problema da
  imagem ou da anotação.
- **`suspeitas_anotacao.csv`**, **`imagens_erros.csv`** (tudo por imagem) e as tabelas
  estatísticas completas.
- Com 2 ou mais modelos, `comparacao_<a>_vs_<b>/` separa imagens **difíceis para todos**
  (provável problema nos dados) das **difíceis só para um modelo**.

Os indícios usam só estatística simples (Spearman, comparação piores × demais) com correção
para múltiplos testes (Benjamini-Hochberg, FDR 5%), para não "achar" padrões por acaso. São
correlações, não causas: use as grades de imagens para confirmar.

Para testar uma hipótese nova (por exemplo, "folhas com mancha erram mais"), adicione a
característica em `leaf_measures/analysis/features.py` (cálculo + uma linha em `FEATURE_INFO`)
e rode com `--recompute-features`.

## Função de perda

Em `loss.type` no YAML:

| Opção | O que mede |
|---|---|
| `huber` (padrão) | Huber nas medidas normalizadas (0–1): erro **absoluto** |
| `mse` | erro quadrático nas medidas normalizadas |
| `huber_log` | Huber entre os **logs** das medidas reais: erro **relativo**, como o RER |
| `mse_log` | MSLE: erro quadrático entre os logs |
| `l1_log` | erro absoluto entre os logs |

Nas versões `_log`, o `huber_beta` também fica em escala log (0,05 ≈ 5% de erro). O exemplo
`configs/experiments/resnet50_rgb_huber_log.yaml` é o baseline trocando só a perda.

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
