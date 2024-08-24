from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import os
import logging
# pylint: disable=import-error
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Configure le niveau de logging
logging.basicConfig(level=logging.INFO)

# Chemins de fichiers
MODEL_PATH = os.path.abspath(
    "C:/Users/leenc/Documents/openclassrooms/p5/api/mlruns/artifacts/use_model.h5")
PICKLE_PATH = os.path.abspath(
    "C:/Users/leenc/Documents/openclassrooms/p5/api/mlruns/artifacts/top_tags.pkl")

# Charger le module USE (Universal Sentence Encoder) depuis TensorFlow Hub
use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

# Charger le modèle de classification


def load_classification_model(model_path):
    try:
        model = load_model(model_path)
        logging.info(f"Modèle de classification chargé depuis {model_path}")
        return model
    except Exception as e:
        logging.error(
            f"Erreur lors du chargement du modèle de classification : {str(e)}")
        raise e


classification_model = load_classification_model(MODEL_PATH)

# Charger les objets BoW+SVD (les étiquettes de classification)


def load_bow_svd_objects(pickle_path):
    try:
        with open(pickle_path, 'rb') as f:
            top_tags = pickle.load(f)
        logging.info(
            f"Objets BoW+SVD chargés avec succès depuis {pickle_path}")
        return top_tags
    except Exception as e:
        logging.error(
            f"Erreur lors du chargement des objets BoW+SVD : {str(e)}")
        raise e


top_tags = load_bow_svd_objects(PICKLE_PATH)

# Transformer un texte en embeddings USE


def transform_text_to_use(text):
    try:
        input_tensor = tf.convert_to_tensor([text], dtype=tf.string)
        embeddings = use_layer(input_tensor).numpy()
        return embeddings
    except Exception as e:
        logging.error(
            f"Erreur lors de la transformation du texte en embeddings USE : {str(e)}")
        raise e

# Endpoint pour les prédictions du modèle USE


@app.route('/predict', methods=['POST'])
def predict_use_tags():
    try:
        # Vérifie que la requête est bien au format JSON
        if not request.is_json:
            return jsonify({'error': 'Invalid request format, JSON expected'}), 400

        # Tente de récupérer le texte de la question depuis la requête JSON
        try:
            question_text = request.json.get('question_text')
        except Exception:
            return jsonify({'error': 'Invalid JSON format'}), 400

        # Vérifie que le texte de la question est bien présent
        if not question_text:
            return jsonify({'error': 'Texte de la question non fourni'}), 400

        # Transforme le texte en vecteur USE
        question_vector_use = transform_text_to_use(question_text)

        # Prédiction des étiquettes à l'aide du modèle de classification
        predicted_tags_probabilities_use = classification_model.predict(
            question_vector_use)[0]

        # Associe chaque tag à sa probabilité
        tag_probabilities = [(tag, float(prob)) for tag, prob in zip(
            top_tags, predicted_tags_probabilities_use)]

        # Trie les tags par probabilité décroissante
        tag_probabilities_sorted = sorted(
            tag_probabilities, key=lambda x: x[1], reverse=True)

        # Limite à un maximum de 5 tags
        top_5_tags = tag_probabilities_sorted[:5]

        # Sépare les tags et leurs probabilités pour l'affichage
        predicted_tag_names_use = [tag for tag, prob in top_5_tags]
        max_prob_tag_use, max_prob_value_use = top_5_tags[0]

        # Retourne les étiquettes prédites (5) et l'étiquette avec la probabilité la plus élevée
        return jsonify({
            'predicted_tags': predicted_tag_names_use,
            'max_probability_tag': max_prob_tag_use,
            'max_probability_value': max_prob_value_use
        })
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {str(e)}")
        return jsonify({'error': str(e)}), 500
