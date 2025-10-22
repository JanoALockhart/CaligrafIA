from datasets.dataset_broker import DatasetBroker
import settings
from callbacks import ValidationLogCallback
from metrics import CharacterErrorRate, WordErrorRate
from model import build_model
import keras
from pathlib import Path


class ModelManager():
    def __init__(self, dataset_broker:DatasetBroker, logger):
        self.dataset_broker = dataset_broker
        self.logger = logger

    def test():
        pass

    def cualitative_matrix():
        pass


    def train(self):
        latest_model_path = Path(str(settings.LAST_CHECKPOINT_PATH))
        if not latest_model_path.exists():
            print("Trained model not found. Creating new model...")
            model = build_model(
                input_shape=(settings.IMG_HEIGHT, settings.IMG_WIDTH, 1),
                alphabet_length=len(self.dataset_broker.get_encoding_function().get_vocabulary())
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
            metrics=[CharacterErrorRate(self.dataset_broker.get_decoding_function()), WordErrorRate(self.dataset_broker.get_decoding_function())],
            run_eagerly=settings.EAGER_EXECUTION
        )

        if settings.DEBUG_MODE:
            model.summary()

        # TRAINING
        val_log_callback = ValidationLogCallback(self.dataset_broker.get_validation_set(), self.dataset_broker.get_decoding_function(), self.logger)
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
            x=self.dataset_broker.get_training_set(),
            epochs=settings.EPOCHS,
            validation_data=self.dataset_broker.get_validation_set(),
            callbacks=[
                val_log_callback,
                metrics_log_callback,
                model_checkpoint_callback,
                latest_checkpoint_callback,
            ],
        )