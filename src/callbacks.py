import tensorflow as tf
import keras
import numpy as np
import settings
import logging

class ValidationLogCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, int_to_char, logger):
        super().__init__()
        self.val_ds = val_ds
        self.int_to_char = int_to_char
        self.logger = logger

    def on_epoch_end(self, epoch: int, logs=None):
        self.logger.info("=" * 100)
        self.logger.info(f"VALIDATION OF EPOCH {epoch}")
        self.logger.info("=" * 100)
        
        
        for batch in self.val_ds:
            x, y = batch
            logits = self.model(x, training=False)
            sparse_true_batch, sparse_pred_batch = self._decode_batch(y, logits)
            self._show_predicted_vs_true(sparse_true_batch, sparse_pred_batch)

    def _decode_batch(self, y_true_batch, y_pred_batch):
        ragged_true_batch = tf.RaggedTensor.from_tensor(y_true_batch, padding=0)
        sparse_true_batch = ragged_true_batch.to_sparse()
        sparse_true_batch = self.int_to_char(sparse_true_batch)

        batch_size = tf.shape(y_pred_batch)[0]
        timesteps = tf.shape(y_pred_batch)[1]
        y_pred_len = timesteps * tf.ones(batch_size, dtype="int32")
        top_paths, probabilities = keras.ops.ctc_decode(y_pred_batch, sequence_lengths=y_pred_len, strategy="greedy")
        y_pred_ctc_decoded = top_paths[0]
        ragged_pred_batch = tf.RaggedTensor.from_tensor(y_pred_ctc_decoded, padding=-1)
        sparse_pred_batch = ragged_pred_batch.to_sparse()
        sparse_pred_batch = self.int_to_char(sparse_pred_batch)

        if settings.DEBUG_MODE:
            print("CALLBACK y_true: ", sparse_true_batch.values)
            print("CALLBACK y_pred: ", sparse_pred_batch.values)

        return sparse_true_batch, sparse_pred_batch

    def _get_text(self, sparse_text):
        dense_text = tf.sparse.to_dense(sparse_text)
        text = tf.strings.reduce_join(dense_text, axis=1)
        if settings.DEBUG_MODE:
            print("CALLBACK dense text: ", dense_text)
            print("CALLBACK joint text: ", text)

        return text

    # TODO: Use tf operations (work with tensors). Copy CER. Check for bug, every val prediction is returning nothing
    def _show_predicted_vs_true(self, sparse_true_batch, sparse_pred_batch, max_to_show=2):
        true_text_batch = self._get_text(sparse_true_batch)
        pred_text_batch = self._get_text(sparse_pred_batch)
        cers = tf.edit_distance(sparse_pred_batch, sparse_true_batch, normalize=True)
        wers = self._calculate_wer(sparse_true_batch, sparse_pred_batch)

        for true_text, pred_text, cer, wer in zip(true_text_batch[:max_to_show], pred_text_batch[:max_to_show], cers[:max_to_show], wers[:max_to_show]):
            self.logger.info(f"True      : {true_text}")
            self.logger.info(f"Predicted : {pred_text}")
            self.logger.info(f"CER       : {cer * 100: .2f}%")
            self.logger.info(f"WER       : {wer * 100: .2f}%")
            self.logger.info("-" * 100)

    def _calculate_wer(self, sparse_true_batch, sparse_pred_batch):
        true_strings = self._get_sparse_strings(sparse_true_batch)
        pred_strings = self._get_sparse_strings(sparse_pred_batch)

        wers = tf.edit_distance(pred_strings, true_strings, normalize=True)

        return wers
    
    def _get_sparse_strings(self, sparse_chars):
        strings = self._get_text(sparse_chars)
        strings = tf.strings.split(strings).to_sparse()

        return strings