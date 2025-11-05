from matplotlib import pyplot as plt
import tensorflow as tf
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

        keras.utils.plot_model(model, settings.MODEL_ARCHITECTURE_FILE_PATH)

        #COMPILE
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss= keras.losses.CTC(),
            metrics=[CharacterErrorRate(self.dataset_broker.get_decoding_function()), WordErrorRate(self.dataset_broker.get_decoding_function())],
            run_eagerly=settings.EAGER_EXECUTION
        )

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
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor = "val_CER",
            patience = 10,
            mode = "min"
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
                early_stopping_callback
            ],
        )
    
    
    def test(self, model_path):
        model = keras.models.load_model(filepath=model_path, compile=False)
        cer_metric = CharacterErrorRate(self.dataset_broker.get_decoding_function())
        wer_metric = WordErrorRate(self.dataset_broker.get_decoding_function())

        for batch in self.dataset_broker.get_test_set():
            x, y = batch
            logits = model.predict(x)
            cer_metric.update_state(y, logits)
            wer_metric.update_state(y, logits)

        cer = cer_metric.result()
        wer = wer_metric.result()

        return cer, wer
    
    # TODO: Refactor plot out
    def qualitative_matrix(self, model_path, side=4):
        model = keras.models.load_model(filepath=model_path, compile=False)
        cer_metric = CharacterErrorRate(self.dataset_broker.get_decoding_function())

        plt.figure(figsize=(18, 9))
        for i, sample in enumerate(self.dataset_broker.get_test_set().unbatch().take(side*side)):
            x, y = sample
            x = tf.expand_dims(x, axis=0)
            y = tf.expand_dims(y, axis=0)

            logits = model.predict(x)
            cer_metric.update_state(y, logits)
            cer = cer_metric.result()
            cer_metric.reset_state()

            img = tf.squeeze(x, axis=0)
            true_text = self.dataset_broker.get_decoding_function()(y)
            true_text = tf.strings.reduce_join(true_text).numpy().decode("utf-8").replace("[UNK]", "")
            pred_text = self._decode_logits(logits)
            plt.subplot(side, side, i+1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"True: {true_text} \n Pred: {pred_text}\n CER: {cer*100: .2f}", fontsize=8)

        plt.savefig(f"{settings.PLOTS_PATH}/cualitative_matrix.png")
        plt.show()

    def _decode_logits(self, logits):
        print(logits.shape)

        input_len = tf.ones(logits.shape[0]) * logits.shape[1]
        top_paths, probabilities = keras.ops.ctc_decode(logits, sequence_lengths=input_len, strategy="greedy")
        y_pred_ctc_decoded = top_paths[0][0]
        pred_string = tf.strings.reduce_join(self.dataset_broker.get_decoding_function()(y_pred_ctc_decoded)).numpy().decode("utf-8").replace("[UNK]", "")
        return pred_string
