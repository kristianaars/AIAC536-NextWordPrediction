import numpy as np
import tensorflow as tf
from keras.src.utils import pad_sequences
import pickle

class NextWordPredictor:

    def __init__(self, nwp_model_model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(nwp_model_model_path)
        self.max_sequence_length = self.extract_max_sequence_length()
        self.tokenizer = self.load_tokenizer(tokenizer_path)

    def load_tokenizer(self, file_path):
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)

    def extract_max_sequence_length(self):
        return self.model.layers[0].get_output_at(0).get_shape().as_list()[1] + 1

    def predict(self, seed_text, n_suggestions=5):
        token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=self.max_sequence_length - 1,
            padding='pre'
        )

        predictions = self.model.predict(token_list)
        top_indexes = np.argpartition(predictions[0], -n_suggestions)[-n_suggestions:]
        return [self.tokenizer.index_word[i] for i in top_indexes]
