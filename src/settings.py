import os
from dotenv import load_dotenv

load_dotenv()
base_path = os.path.dirname(__file__)

IAM_PATH = os.getenv("IAM_PATH")
CVL_PATH = os.getenv("CVL_PATH")
RIMES_PATH = os.getenv("RIMES_PATH")

DEBUG_MODE = bool(os.getenv("DEBUG_MODE"))

EPOCHS = int(os.getenv("EPOCHS")) if not DEBUG_MODE else int(os.getenv("DEBUG_EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE")) if not DEBUG_MODE else int(os.getenv("DEBUG_BATCH_SIZE"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT")) if not DEBUG_MODE else float(os.getenv("DEBUG_TRAIN_SPLIT"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT")) if not DEBUG_MODE else float(os.getenv("DEBUG_VAL_SPLIT"))

IMG_HEIGHT = int(os.getenv("IMG_HEIGHT"))
IMG_WIDTH = int(os.getenv("IMG_WIDTH"))

EAGER_EXECUTION = False if not DEBUG_MODE else bool(os.getenv("EAGER_EXECUTION"))

TEST_IMG_PATH = os.getenv("TEST_IMG_PATH")

HISTORY_PATH = os.path.abspath(os.path.join(base_path, os.getenv("HISTORY_PATH")))
BEST_CHECKPOINT_PATH = os.path.abspath(os.path.join(base_path, os.getenv("BEST_CHECKPOINT_PATH")))
LAST_CHECKPOINT_PATH = os.path.abspath(os.path.join(base_path, os.getenv("LAST_CHECKPOINT_PATH")))
VALIDATION_LOG_PATH = os.path.abspath(os.path.join(base_path, os.getenv("VALIDATION_LOG_PATH")))
PLOTS_PATH = os.path.abspath(os.path.join(base_path, os.getenv("PLOTS_PATH")))
SAVED_MODELS_PATH = os.path.abspath(os.path.join(base_path, os.getenv("SAVED_MODELS_PATH")))
TRAINING_METRICS_FILE_PATH = os.path.abspath(os.path.join(base_path, os.getenv("TRAINING_METRICS_FILE_PATH")))
TEST_METRICS_FILE_PATH = os.path.abspath(os.path.join(base_path, os.getenv("TEST_METRICS_FILE_PATH")))
DATASETS_INFO_FILE_PATH = os.path.abspath(os.path.join(base_path, os.getenv("DATASETS_INFO_FILE_PATH")))
