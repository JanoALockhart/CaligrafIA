import logging
from pathlib import Path
from datasets.dataset_broker import DatasetBrokerImpl
from datasets.iam.iam_dataset_builder import IAMDatasetBuilder
from metrics import CharacterErrorRate, WordErrorRate
import settings
import matplotlib.pyplot as plt
import keras
import numpy as np
from datasets.iam.iam_dataloader import IAMLineDataloader

from model import build_model
from callbacks import ValidationLogCallback

def main():
    input_shape = (32, 256, 1)

    logger = configure_validation_logger()
    dataset_broker = configure_datasets()

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

    # MODEL
    latest_model_path = Path(str(settings.LAST_CHECKPOINT_PATH))
    if not latest_model_path.exists():
        print("Trained model not found. Creating new model...")
        model = build_model(
            input_shape=(settings.IMG_HEIGHT, settings.IMG_WIDTH, 1), 
            alphabet_length=len(dataset_broker.get_encoding_function().get_vocabulary())
        )
    else:
        print("Trained model found. Loading model...")
        model = keras.models.load_model(
            filepath=settings.LAST_CHECKPOINT_PATH,
            compile=False
        )

    #COMPILE
    model.compile(
        optimizer=keras.optimizers.Adam(), 
        loss= keras.losses.CTC(),
        metrics=[CharacterErrorRate(dataset_broker.get_decoding_function()), WordErrorRate(dataset_broker.get_decoding_function())],
        run_eagerly=settings.EAGER_EXECUTION
    )
    
    if settings.DEBUG_MODE:
        model.summary()

    # TRAINING
    val_log_callback = ValidationLogCallback(dataset_broker.get_validation_set(), dataset_broker.get_decoding_function(), logger)
    metrics_log_callback = keras.callbacks.CSVLogger(settings.HISTORY_PATH, append=True)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=settings.BEST_CHECKPOINT_PATH,
        monitor="val_CER",
        verbose=1,
        save_best_only=True,
        mode="min"
    )
    latest_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=settings.LAST_CHECKPOINT_PATH,
        save_best_only=False
    )

    history = model.fit(
        x=dataset_broker.get_training_set(), 
        epochs=settings.EPOCHS, 
        validation_data=dataset_broker.get_validation_set(),
        callbacks=[
            val_log_callback,
            metrics_log_callback,
            model_checkpoint_callback,
            latest_checkpoint_callback,
        ],
    )

def configure_datasets():
    dataset_broker = DatasetBrokerImpl(
        train_split_per=settings.TRAIN_SPLIT,
        val_split_per=settings.VAL_SPLIT,
        img_height=settings.IMG_HEIGHT,
        img_width=settings.IMG_WIDTH,
        batch_size=settings.BATCH_SIZE,
    )

    iam_loader = IAMLineDataloader(settings.IAM_PATH)
    iam_builder = IAMDatasetBuilder(iam_loader)
    dataset_broker.register_dataset_builder(iam_builder)

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