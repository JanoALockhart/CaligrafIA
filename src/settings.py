import os
from dotenv import load_dotenv

load_dotenv()

IAM_PATH = os.getenv("IAM_PATH")
CVL_PATH = os.getenv("CVL_PATH")
RIMES_PATH = os.getenv("RIMES_PATH")

DEBUG_MODE = bool(os.getenv("DEBUG_MODE"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE")) if not DEBUG_MODE else int(os.getenv("DEBUG_BATCH_SIZE"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT")) if not DEBUG_MODE else float(os.getenv("DEBUG_TRAIN_SPLIT"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT")) if not DEBUG_MODE else float(os.getenv("DEBUG_VAL_SPLIT"))