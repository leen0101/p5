from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import pickle

app = Flask(__name__)

# Modèle USE à partir de TensorFlow Hub
use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

# Modèle de classification pré-entraîné après USE
classification_model = tf.keras.models.load_model(
    "../notebooks/model/use_model.keras")

# Objets du modèle BoW+SVD (pour obtenir les tags)


def load_bow_svd_objects():
    with open('../notebooks/model/top_tags.pkl', 'rb') as f:
        top_tags = pickle.load(f)
    return top_tags


top_tags = load_bow_svd_objects()


def transform_text_to_use(text):
    input_tensor = tf.convert_to_tensor([text], dtype=tf.string)
    embeddings = use_layer(input_tensor).numpy()
    return embeddings

# Endpoint Flask pour le modèle USE


@app.route('/predict_use_tags', methods=['POST'])
def predict_use_tags():
    try:
        question_text = request.json.get('question_text')
        if not question_text:
            return jsonify({'error': 'No question text provided'}), 400

        # Transformation du texte en vecteur USE
        question_vector_use = transform_text_to_use(question_text)

        # Mdèle de classification pour prédire les tags
        predicted_tags_probabilities_use = classification_model.predict(
            question_vector_use)

        # Index du tag avec la probabilité maximale
        max_prob_index_use = predicted_tags_probabilities_use.argmax()
        max_prob_tag_use = top_tags[max_prob_index_use]

        # Tags prédits ayant une probabilité > 0.01
        predicted_tag_names_use = [top_tags[i] for i in range(
            len(top_tags)) if predicted_tags_probabilities_use[0][i] > 0.01]

        max_prob_value_use = float(
            predicted_tags_probabilities_use[0][max_prob_index_use])

        # Tags prédits et le tag avec la probabilité maximale
        return jsonify({
            'predicted_tags': predicted_tag_names_use,
            'max_probability_tag': max_prob_tag_use,
            'max_probability_value': max_prob_value_use
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
