from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

model_path = os.path.abspath(
    "C:/Users/leenc/Documents/openclassrooms/p5/api/mlruns/artifacts/use_model.h5")
try:
    classification_model = tf.keras.models.load_model(model_path)
    logging.info(f"Classification model loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"Error loading classification model: {str(e)}")
    raise e


def load_bow_svd_objects():
    try:
        pkl_path = os.path.abspath(
            'C:/Users/leenc/Documents/openclassrooms/p5/api/mlruns/artifacts/top_tags.pkl')
        with open(pkl_path, 'rb') as f:
            top_tags = pickle.load(f)
        logging.info(f"BoW+SVD objects loaded successfully from {pkl_path}")
        return top_tags
    except Exception as e:
        logging.error(f"Error loading BoW+SVD objects: {str(e)}")
        raise e


top_tags = load_bow_svd_objects()


def transform_text_to_use(text):
    try:
        input_tensor = tf.convert_to_tensor([text], dtype=tf.string)
        embeddings = use_layer(input_tensor).numpy()
        return embeddings
    except Exception as e:
        logging.error(f"Error transforming text to USE embeddings: {str(e)}")
        raise e


@app.route('/predict', methods=['POST'])
def predict_use_tags():
    try:
        question_text = request.json.get('question_text')
        if not question_text:
            return jsonify({'error': 'No question text provided'}), 400

        question_vector_use = transform_text_to_use(question_text)

        predicted_tags_probabilities_use = classification_model.predict(
            question_vector_use)

        max_prob_index_use = predicted_tags_probabilities_use.argmax()
        max_prob_tag_use = top_tags[max_prob_index_use]

        predicted_tag_names_use = [top_tags[i] for i in range(len(top_tags))
                                   if predicted_tags_probabilities_use[0][i] > 0.01]

        max_prob_value_use = float(
            predicted_tags_probabilities_use[0][max_prob_index_use])

        return jsonify({
            'predicted_tags': predicted_tag_names_use,
            'max_probability_tag': max_prob_tag_use,
            'max_probability_value': max_prob_value_use
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
