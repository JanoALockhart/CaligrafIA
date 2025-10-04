import keras
import tensorflow as tf
import settings

class CharacterErrorRate(keras.metrics.Metric):
    def __init__(self, int_to_char, name="CER", **kwargs):
        super().__init__(name=name, **kwargs)
        self.int_to_char = int_to_char
        self.editdistance = self.add_weight(name="ed", initializer="zeros")
        self.total_chars = self.add_weight(name="tc", initializer="zeros")

    def update_state(self, y_true_batch, y_pred_batch, sample_weight=None):
        ragged_true_batch = tf.RaggedTensor.from_tensor(y_true_batch, padding=0)
        sparse_true_batch = ragged_true_batch.to_sparse()
        sparse_true_batch = self.int_to_char(sparse_true_batch)
        
        y_pred_len = tf.fill(tf.shape(y_pred_batch)[0], tf.shape(y_pred_batch)[1])
        top_paths, probabilities = keras.ops.ctc_decode(y_pred_batch, sequence_lengths=y_pred_len, strategy="greedy")
        y_pred_ctc_decoded = top_paths[0]
        ragged_pred_batch = tf.RaggedTensor.from_tensor(y_pred_ctc_decoded, padding=-1)
        sparse_pred_batch = ragged_pred_batch.to_sparse()
        sparse_pred_batch = self.int_to_char(sparse_pred_batch)

        errors = tf.edit_distance(sparse_pred_batch, sparse_true_batch, normalize=False) # shape = (8,)
        errors = tf.math.reduce_sum(errors)
        length = ragged_true_batch.row_lengths() # shape = (8,)
        length = tf.math.reduce_sum(length)

        if settings.DEBUG_MODE:
            print("CER true: ", y_true_batch.shape)
            print("CER pred shape: ", y_pred_batch.shape)
            print("CER true example: ", y_true_batch[0])
            print("CER Logits pred: ", y_pred_ctc_decoded)
            print("CER sparse true values: ", sparse_true_batch.values)
            print("CER sparse pred values", sparse_pred_batch.values)
            print("CER errors", errors)
            print("CER length", length)

            
        self.editdistance.assign_add(errors)
        self.total_chars.assign_add(length)
            
    def result(self):        
        return tf.math.divide_no_nan(self.total_chars, self.editdistance)

    def reset_state(self):
        self.editdistance.assign(0.0)
        self.total_chars.assign(0.0)

# TODO: Implement WER and Phrase Acc
class WordErrorRate(keras.metrics.Metric):
    def __init__(self, int_to_char, name="WER", **kwargs):
        super().__init__(name=name, **kwargs)
        self.int_to_char = int_to_char
        self.editdistance = self.add_weight(name="ed", initializer="zeros")
        self.total_words = self.add_weight(name="tw", initializer="zeros")

    def update_state(self, y_true_batch, y_pred_batch, sample_weight=None):
        ragged_true_batch = tf.RaggedTensor.from_tensor(y_true_batch, padding=0)
        sparse_true_batch = ragged_true_batch.to_sparse()
        sparse_true_batch = self.int_to_char(sparse_true_batch)
        sparse_string_true_batch = self._get_sparse_strings(sparse_true_batch)

        y_pred_len = tf.fill(tf.shape(y_pred_batch)[0], tf.shape(y_pred_batch)[1])
        top_paths, probabilities = keras.ops.ctc_decode(y_pred_batch, sequence_lengths=y_pred_len, strategy="greedy")
        y_pred_ctc_decoded = top_paths[0]
        ragged_pred_batch = tf.RaggedTensor.from_tensor(y_pred_ctc_decoded, padding=-1)
        sparse_pred_batch = ragged_pred_batch.to_sparse()
        sparse_pred_batch = self.int_to_char(sparse_pred_batch)
        sparse_string_pred_batch = self._get_sparse_strings(sparse_pred_batch)

        errors = tf.edit_distance(sparse_string_pred_batch, sparse_string_true_batch, normalize=False)
        errors = tf.math.reduce_sum(errors)
        length = tf.RaggedTensor.from_sparse(sparse_string_true_batch).row_lengths() # shape = (8,)
        length = tf.math.reduce_sum(length)

        if settings.DEBUG_MODE:
            print("WER true string sparse: ", sparse_string_true_batch.values)
            print("WER pred string sparse: ", sparse_string_pred_batch.values)
            print("WER errors", errors)
            print("WER length", length)

        self.editdistance.assign_add(errors)
        self.total_words.assign_add(length)

    def result(self):        
        return tf.math.divide_no_nan(self.total_words, self.editdistance)

    def reset_state(self):
        self.editdistance.assign(0.0)
        self.total_words.assign(0.0)

    def _get_sparse_strings(self, sparse_chars):
        dense_text = tf.sparse.to_dense(sparse_chars)
        strings = tf.strings.reduce_join(dense_text, axis=1)
        strings = tf.strings.split(strings).to_sparse()

        return strings


