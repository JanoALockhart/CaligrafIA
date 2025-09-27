import tensorflow as tf
import keras
import numpy as np
import settings

class ValidationLogCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, int_to_char):
        super().__init__()
        self.val_ds = val_ds
        self.int_to_char = int_to_char

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []

        for batch in self.val_ds:
            x, y = batch
            logits = self.model(x)
            batch_predicted_text = self._decode_logits(logits)
            batch_true_text = self._decode_labels(y)
            self._show_predicted_vs_true(batch_predicted_text, batch_true_text)

        
    def _decode_logits(self, logits):
        batch_predicted_texts = []
        input_len = np.ones(logits.shape[0]) * logits.shape[1]
        top_paths, probabilities = keras.ops.ctc_decode(inputs=logits, sequence_lengths=input_len, strategy="greedy")
        batch_results = top_paths[0]

        if settings.DEBUG_MODE:
            print("Logits: ", logits.shape)
            print("CTC Decoded: ", batch_results.shape)
            print("Decode[0]: ", batch_results[0].shape)
        
        for encoded_line in batch_results:
            decoded_line_chars = self.int_to_char(encoded_line)
            text = tf.strings.reduce_join(decoded_line_chars).numpy().decode("utf-8")
            batch_predicted_texts.append(text)
        
        return batch_predicted_texts
    
    def _decode_labels(self, y_true_batch):
        batch_true_texts = []
        if settings.DEBUG_MODE:
            print("y_true_batch: ", y_true_batch.shape)

        for label in y_true_batch:
            decoded_line_char = self.int_to_char(label)
            text = tf.strings.reduce_join(decoded_line_char).numpy().decode("utf-8")
            batch_true_texts.append(text)

        return batch_true_texts
    
    def _show_predicted_vs_true(self, batch_predicted_text, batch_true_text):
        for pred_text, true_text in zip(batch_predicted_text, batch_true_text):
            print(f"True      : {true_text}")
            print(f"Predicted : {pred_text}")
            print("-" * 100)
        