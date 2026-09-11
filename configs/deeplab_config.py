NUM_CLASSES = 3
BACKBONE = "xception"
OUTPUT_STRIDE = 16
SYNC_BN = None
FREEZE_BN = True

LEARNING_RATE = 0.00002
MOMENTUM = 0.9
WEIGHT_DECAY = 4e-5

ITERS = 65000
POWER = 0.9

VAL_INTERVAL = 500
BATCH_SIZE = 2

NUM_WORKERS = 4
PIN_MEMORY = True

MEASURE_WEIGHTS_VALUES = [1.0, 1.0, 1.0, 5.0]  # length, width, perimeter, area
SEG_LOSS_WEIGHT = 1.0
MEASURES_LOSS_WEIGHT = 10.0

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512

# --------------------------------------------------------------------
ROOT = r"Leaf on Stem Image Dataset Beans (LSID-Beans)\LSID-Beans"
SPLIT_DIR = r"Leaf on Stem Image Dataset Beans (LSID-Beans)\cross-validation\cv_1\split_1"

TRAIN_CSV = SPLIT_DIR + r"\train.csv"
VAL_CSV = SPLIT_DIR + r"\val.csv"
TEST_CSV = SPLIT_DIR + r"\test.csv"

CHECKPOINT_PATH = "best_model_deeplab_multitask.pth"
