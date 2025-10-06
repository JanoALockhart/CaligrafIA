import os
from dotenv import load_dotenv

load_dotenv()

IAM_PATH = os.getenv("IAM_PATH")
CVL_PATH = os.getenv("CVL_PATH")
RIMES_PATH = os.getenv("RIMES_PATH")

DEBUG_MODE = bool(os.getenv("DEBUG_MODE"))

EPOCHS = int(os.getenv("EPOCHS")) if not DEBUG_MODE else int(os.getenv("DEBUG_EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE")) if not DEBUG_MODE else int(os.getenv("DEBUG_BATCH_SIZE"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT")) if not DEBUG_MODE else float(os.getenv("DEBUG_TRAIN_SPLIT"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT")) if not DEBUG_MODE else float(os.getenv("DEBUG_VAL_SPLIT"))

EAGER_EXECUTION = False if not DEBUG_MODE else bool(os.getenv("EAGER_EXECUTION"))

TEST_IMG_PATH = os.getenv("TEST_IMG_PATH")

HISTORY_PATH = os.getenv("HISTORY_PATH")
BEST_CHECKPOINT_PATH = os.getenv("BEST_CHECKPOINT_PATH")
LAST_CHECKPOINT_PATH = os.getenv("LAST_CHECKPOINT_PATH")
VALIDATION_LOG_PATH = os.getenv("VALIDATION_LOG_PATH")