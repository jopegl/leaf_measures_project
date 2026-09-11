"""
Leitura dos arquivos de split (train.csv / val.csv / test.csv).

IMPORTANTE (leia antes de mexer):
- 'train'  -> usado para treinar os pesos do modelo.
- 'val'    -> usado DURANTE o treino, a cada VAL_INTERVAL iterações, para
              decidir qual checkpoint é o melhor. É nele que a seleção de
              modelo acontece.
- 'test'   -> NUNCA é olhado durante o treino. Só é usado uma única vez,
              depois de já ter escolhido o checkpoint final, para reportar
              a métrica real do modelo. Ver evaluate_test.py.

Misturar val e test (ou usar test para escolher o checkpoint) contamina a
métrica final reportada e some com o objetivo de ter um teste imparcial.
"""
import os
import pandas as pd

from common.annotation import strip_known_suffix


def load_split_ids(split_file: str) -> set:
    ids = set()

    if split_file.endswith(".csv"):
        df = pd.read_csv(split_file, header=None)
        for item in df[0]:
            ids.add(strip_known_suffix(str(item)))
    else:
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    line = os.path.splitext(line)[0]
                    ids.add(strip_known_suffix(line))

    return ids
