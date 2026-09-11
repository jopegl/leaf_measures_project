# leaf_measures_project (mesclado)

Versão modularizada dos dois scripts de treino (ResNetRegressor e DeepLab
multi-tarefa), com a correção do fluxo de validação/teste.

> **Nota sobre esta mesclagem:** este projeto une `leaf_measures_project`
> (a versão modularizada) com o repositório original `leaf-deeplab-copia`.
> O `leaf_measures_project` original dependia de um arquivo,
> `deeplab_just_measures.py`, que não estava incluído nele — o README dizia
> explicitamente "basta colocar esse arquivo na raiz do projeto". Esse
> arquivo e as suas próprias dependências (`aspp.py`, `decoder.py`,
> `backbone/`, `sync_batchnorm/`) foram copiados do `leaf-deeplab-copia`
> para a raiz deste projeto, então agora ele roda sem passos manuais
> extras. Os scripts antigos do `leaf-deeplab-copia` que foram substituídos
> pela versão modularizada (`test.py`, `test-with_measures.py`,
> `train_just_measures.py`, `resnet_train.py`, `deeplab.py`,
> `deeplab_seg.py`) ficaram guardados em `legacy_original_scripts/` apenas
> como referência histórica — não use-os para treinar/avaliar, use os
> scripts na raiz.

## O que mudou

Antes, os dois scripts avaliavam periodicamente no `test.csv` e usavam esse
mesmo resultado pra decidir qual checkpoint salvar como "o melhor". Isso
contamina a métrica final — o teste deixa de ser uma estimativa imparcial,
porque foi usado repetidamente para guiar decisões (ver conversa anterior
sobre bias/variance e overfitting no teste).

Agora:

- `train_*.py` usa `train.csv` para treinar e **`val.csv` para escolher o
  checkpoint**. `test.csv` nunca é carregado nesses scripts.
- `evaluate_test.py` carrega o checkpoint final e roda **uma única vez**
  sobre `test.csv`, produzindo a métrica final e imparcial.

## Estrutura de pastas

```
leaf_measures_project/
├── README.md
├── LICENSE
├── common/
│   ├── __init__.py
│   ├── annotation.py      # parsing do XML de anotação
│   ├── splits.py          # leitura de train/val/test.csv
│   ├── normalization.py   # min-max das medidas (length, width, perimeter, area)
│   ├── datasets.py        # LSIDBeansMeasuresDataset e LSIDBeansDataset
│   ├── metrics.py         # calculate_measure_metrics, calculate_miou
│   ├── losses.py          # weighted_measures_loss (mse/huber)
│   └── checkpoint.py      # save_checkpoint / load_checkpoint_if_exists
├── models/
│   ├── __init__.py
│   └── resnet_regressor.py
├── configs/
│   ├── __init__.py
│   ├── resnet_config.py   # hiperparâmetros + caminhos (TRAIN/VAL/TEST_CSV)
│   └── deeplab_config.py
├── deeplab_just_measures.py   # definição do modelo DeepLab (trazido do leaf-deeplab-copia)
├── aspp.py                    # dependência de deeplab_just_measures.py
├── decoder.py                 # dependência de deeplab_just_measures.py
├── backbone/                  # dependência de deeplab_just_measures.py (resnet/xception/drn/mobilenet)
├── sync_batchnorm/            # dependência de aspp.py/decoder.py/backbone/deeplab_just_measures.py
├── train_resnet_measures.py    # treino do ResNetRegressor
├── train_deeplab_multitask.py  # treino do DeepLab (segmentação + medidas)
├── evaluate_test.py             # avaliação final única no teste
└── legacy_original_scripts/     # scripts pré-modularização, mantidos só como referência
    ├── test.py
    ├── test-with_measures.py
    ├── train_just_measures.py
    ├── resnet_train.py
    ├── deeplab.py
    └── deeplab_seg.py
```

`deeplab_just_measures.py` (a definição do modelo DeepLab) e suas
dependências (`aspp.py`, `decoder.py`, `backbone/`, `sync_batchnorm/`) vêm
do `leaf-deeplab-copia` e já estão na raiz deste projeto — não é preciso
nenhum passo manual, `train_deeplab_multitask.py` e `evaluate_test.py`
importam tudo diretamente.

## Estrutura de dados esperada

```
Leaf on Stem Image Dataset Beans (LSID-Beans)/
├── LSID-Beans/
│   └── <leaf_folder>/
│       ├── images/<id>.jpg
│       ├── annotation/<id>_annotation.xml
│       └── segmentation/<id>_segmentation.raw   (só usado pelo DeepLab)
└── cross-validation/
    └── cv_1/
        └── split_1/
            ├── train.csv
            ├── val.csv      <-- o que você já tinha preparado
            └── test.csv
```

## Como rodar

```bash
# 1. Treinar o regressor ResNet (val.csv guia a seleção de checkpoint)
python train_resnet_measures.py

# 2. Ou treinar o DeepLab multi-tarefa
python train_deeplab_multitask.py

# 3. Só no final, avaliação imparcial no teste
python evaluate_test.py --model resnet
python evaluate_test.py --model deeplab
```

## Se você já tem checkpoints treinados com o método antigo

Eles ainda são válidos como pesos de modelo — o problema estava só em *como*
o "melhor" checkpoint foi escolhido, não nos pesos em si. Se quiser, dá pra
comparar os candidatos que você já tem contra `val.csv` (via
`evaluate_test.py` adaptado, ou um script parecido) para confirmar se a
escolha do "definitivo" muda, antes de reportar qualquer número em
publicações.
