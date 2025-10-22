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
import numpy as np
from datasets.iam.iam_dataloader import IAMLineDataloader


def main():
    logger = configure_validation_logger()
    dataset_broker = configure_datasets()
    model_manager = ModelManager(dataset_broker, logger)

    debug(dataset_broker)

    model_manager.train()

def debug(dataset_broker):
    if settings.DEBUG_MODE:
        print("--- DEBUG MODE ACTIVE ---")

    if settings.DEBUG_MODE:
        vocab = dataset_broker.get_encoding_function().get_vocabulary()
        print("Char classes:", vocab, "Len: ", len(vocab))

    if settings.DEBUG_MODE:
        train_ds = dataset_broker.get_training_set()
        print("Splits Batched:  ", train_ds.cardinality(), dataset_broker.get_validation_set().cardinality(), dataset_broker.get_test_set().cardinality())

        for (sample, label) in train_ds.take(1):
            print("DS sample shape: ", sample.numpy().shape)
            print("Max, min values in sample: ", np.max(sample[0].numpy()), np.min(sample[0].numpy()))
            print("y_true: ", label.numpy())
            #plt.imshow(sample[0])
            #plt.title(tf.strings.reduce_join(int_to_char(label[0])).numpy().decode("UTF-8"))
            #plt.show()

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
    print(log_path)
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