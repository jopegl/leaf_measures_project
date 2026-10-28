BACKBONE = "resnet50"
PRETRAINED = True
FREEZE_BACKBONE_EPOCHS = 2  # nº de validações com backbone congelado antes do fine-tuning

NUM_OUTPUTS = 4  # length, width, perimeter, area

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

ITERS = 45000
VAL_INTERVAL = 500
BATCH_SIZE = 4
# 
NUM_WORKERS = 4
PIN_MEMORY = True

MEASURE_WEIGHTS_VALUES = [1.0, 1.0, 1.0, 5.0]  # length, width, perimeter, area

REGRESSION_LOSS = "huber"  # 'mse' ou 'huber'
HUBER_BETA = 0.1

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512


ROOT = r"Leaf on Stem Image Dataset Beans (LSID-Beans)\LSID-Beans"

SPLIT_DIR = r"Leaf on Stem Image Dataset Beans (LSID-Beans)\cross-validation\cv_1\split_1"

TRAIN_CSV = SPLIT_DIR + r"\train.csv"
VAL_CSV = SPLIT_DIR + r"\val.csv"
TEST_CSV = SPLIT_DIR + r"\test.csv"

CHECKPOINT_PATH = "best_model_resnet_measures.pth"
