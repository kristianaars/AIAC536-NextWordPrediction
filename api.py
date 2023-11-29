import json

from flask import Flask, jsonify, request

from NextWordPredictor import NextWordPredictor

next_word_predictor = NextWordPredictor('./models/model_LSTM_128_NGRAM_3_SENTENCE_1000.h5',
                                        './models/tokenizer_LSTM_128_NGRAM_3_SENTENCE_1000.pickle')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_next_word():
    body = json.loads(request.data)
    sentence = body['sentence']
    n_word_suggestions = body['n_word_suggestions']

    suggestions = next_word_predictor.predict(sentence, n_word_suggestions)

    return jsonify({'suggestions': suggestions}), 200


if __name__ == '__main__':
    app.run(port=8080)
