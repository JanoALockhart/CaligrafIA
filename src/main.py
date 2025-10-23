import argparse
import logging
from pathlib import Path
from datasets.cvl.cvl_dataloader import CVLLineDataloader
from datasets.cvl.cvl_dataset_builder import CVLDatasetBuilder
from datasets.dataset_broker import DatasetBrokerImpl
from datasets.iam.iam_dataset_builder import IAMDatasetBuilder
from datasets.rimes.rimes_dataloader import RIMESWordsDataloader
from datasets.rimes.rimes_dataset_builder import RIMESDatasetBuilder
from model_manager import ModelManager
import settings
from datasets.iam.iam_dataloader import IAMLineDataloader

TRAIN = "train"
TEST = "test"
INFO = "info"
MATRIX = "matrix"

def main():
    args = get_command_args()
    logger = configure_validation_logger()
    dataset_broker = configure_datasets()
    model_manager = ModelManager(dataset_broker, logger)

    
    if args.mode == TRAIN:
        model_manager.train()
    
    elif args.mode == TEST:
        model_path = f"{settings.SAVED_MODELS_PATH}{args.load}"
        cer, wer = model_manager.test(model_path)

        with open(settings.TEST_METRICS_FILE_PATH, "w") as file:
            file.write(f"--- Metrics on the Test Set --- \n")
            file.write(f"Model: {model_path}\n")
            file.write(f"Test CER: {cer*100: .2f}%\n")
            file.write(f"Test WER: {wer*100: .2f}%\n")
    
    elif args.mode == INFO:
        dataset_info =  dataset_broker.get_datasets_info()
        with open(settings.INFO_FILE_PATH, "w") as file:
            file.write(dataset_info)
    
    elif args.mode == MATRIX:
        model_path = f"{settings.SAVED_MODELS_PATH}{args.load}"
        model_manager.qualitative_matrix(model_path)


def get_command_args():
    parser = argparse.ArgumentParser(description="Training and evaluation of deep learning model.")
    
    parser.add_argument("-m","--mode", required=True, choices=[TRAIN, TEST, INFO, MATRIX])
    parser.add_argument("-l", "--load", required=False)

    return parser.parse_args()

def configure_datasets():
    dataset_broker = DatasetBrokerImpl(
        train_split_per=settings.TRAIN_SPLIT,
        val_split_per=settings.VAL_SPLIT,
        img_height=settings.IMG_HEIGHT,
        img_width=settings.IMG_WIDTH,
        batch_size=settings.BATCH_SIZE,
        data_augmentation=True
    )

    iam_loader = IAMLineDataloader(settings.IAM_PATH)
    iam_builder = IAMDatasetBuilder(iam_loader)
    dataset_broker.register_dataset_builder(iam_builder)

    rimes_loader = RIMESWordsDataloader(settings.RIMES_PATH)
    rimes_builder = RIMESDatasetBuilder(rimes_loader)
    dataset_broker.register_dataset_builder(rimes_builder)

    cvl_loader = CVLLineDataloader(settings.CVL_PATH)
    cvl_builder = CVLDatasetBuilder(cvl_loader)
    dataset_broker.register_dataset_builder(cvl_builder)

    #Register more datasets builders here
    
    dataset_broker.sample_datasets()
    return dataset_broker

def configure_validation_logger():
    log_path = Path(str(settings.VALIDATION_LOG_PATH))

    if not log_path.exists():
        log_path.touch()

    logging.basicConfig(
        level=logging.INFO,
        filename=settings.VALIDATION_LOG_PATH,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger = logging.getLogger()
    return logger

if __name__ == "__main__":
    main()