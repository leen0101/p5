from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import os
import logging
import tensorflow as tf
# pylint: disable=import-error
from tensorflow.keras.models import load_model  # pylint: disable=no-name-in-module

app = Flask(__name__)

# Configure le niveau de logging
logging.basicConfig(level=logging.INFO)

# Charge le module USE (Universal Sentence Encoder) depuis TensorFlow Hub
use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

# Définit le chemin du modèle de classification
model_path = os.path.abspath(
    "C:/Users/leenc/Documents/openclassrooms/p5/api/mlruns/artifacts/use_model.h5")

# Charge le modèle de classification
try:
    classification_model = load_model(model_path)
    logging.info(f"Modèle de classification chargé depuis {model_path}")
except Exception as e:
    logging.error(f"Erreur modèle de classification : {str(e)}")
    raise e

# Objets BoW+SVD


def load_bow_svd_objects():
    try:
        pkl_path = os.path.abspath(
            'C:/Users/leenc/Documents/openclassrooms/p5/api/mlruns/artifacts/top_tags.pkl')
        with open(pkl_path, 'rb') as f:
            top_tags = pickle.load(f)
        logging.info(f"Objets BoW+SVD chargés avec succès depuis {pkl_path}")
        return top_tags
    except Exception as e:
        logging.error(
            f"Erreur lors du chargement des objets BoW+SVD : {str(e)}")
        raise e


# Charge les objets BoW+SVD (les étiquettes de classification)
top_tags = load_bow_svd_objects()

# Texte en embeddings USE


def transform_text_to_use(text):
    try:
        input_tensor = tf.convert_to_tensor([text], dtype=tf.string)
        embeddings = use_layer(input_tensor).numpy()
        return embeddings
    except Exception as e:
        logging.error(
            f"Erreur lors de la transformation du texte en embeddings USE : {str(e)}")
        raise e

# Endpoint prédictions du modèle USE


@app.route('/predict', methods=['POST'])
def predict_use_tags():
    try:
        question_text = request.json.get('question_text')
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


# Lancement de l'application Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
