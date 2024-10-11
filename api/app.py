from flask import Flask, request, jsonify
import mlflow
import tensorflow as tf
import joblib
import logging
import os
from mlflow.tracking import MlflowClient

# Configure le logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "f8bc1d91ca98.json"

# Configure MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI",
                         "https://mlflowp51-975919512217.us-central1.run.app")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Text_Processing_Experiment")

client = MlflowClient()

# Trouve le run ID le plus récent basé sur le nom de l'artefact


def find_most_recent_run_id_by_artifact(artifact_name):
    """Recherche le run ID associé à un artefact spécifique et retourne le plus récent."""
    experiment = client.get_experiment_by_name("Text_Processing_Experiment")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        for artifact in artifacts:
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(
                    run.info.run_id, artifact.path)
                for sub_artifact in sub_artifacts:
                    if sub_artifact.path.endswith(artifact_name):
                        logging.info("Run trouvé: " + run.info.run_id +
                                     artifact_name)
                        return run.info.run_id
            else:
                if artifact.path.endswith(artifact_name):
                    logging.info("Run trouvé: "+run.info.run_id+"pour " +
                                 artifact_name)
                    return run.info.run_id
    logging.warning("Aucun run trouvé pour "+artifact_name)
    return None


# ID des runs trouvés automatiquement
vectorizer_run_id = find_most_recent_run_id_by_artifact("vectorizer.pkl")
svd_run_id = find_most_recent_run_id_by_artifact("svd.pkl")
top_tags_run_id = find_most_recent_run_id_by_artifact("top_tags.pkl")
model_run_id = find_most_recent_run_id_by_artifact("bow_svd_model.h5")

if not all([vectorizer_run_id, svd_run_id, top_tags_run_id, model_run_id]):
    raise Exception("Impossible de trouver les artefacts nécessaires.")

# Télécharge les artefacts en fonction des `run_id` trouvés


def download_artifact(run_id, artifact_name):
    try:
        path = mlflow.artifacts.download_artifacts(
            "runs:/"+run_id+"/"+artifact_name)
        logging.info("Artefact " + artifact_name+" téléchargé avec succès.")
        return path
    except Exception as e:
        logging.error("Erreur lors du téléchargement de "+artifact_name+": "+e)
        raise


vectorizer_path = download_artifact(vectorizer_run_id, "vectorizer.pkl")
svd_path = download_artifact(svd_run_id, "svd.pkl")
top_tags_path = download_artifact(top_tags_run_id, "top_tags.pkl")
model_path = download_artifact(model_run_id, "bow_svd_model.h5")

# Charge les artefacts
with open(vectorizer_path, 'rb') as f:
    vectorizer = joblib.load(f)

with open(svd_path, 'rb') as f:
    svd = joblib.load(f)

with open(top_tags_path, 'rb') as f:
    top_tags = joblib.load(f)

# Charge le modèle
try:
    model = tf.keras.models.load_model(model_path)
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error("Erreur lors du chargement du modèle : "+e)
    raise

# Transforme le texte en vecteur BoW + SVD


def transform_text_to_bow_svd(text):
    X_bow = vectorizer.transform([text])  # Convertit en vecteur BoW
    X_svd = svd.transform(X_bow)  # Réduit la dimensionnalité avec SVD
    return X_svd

# Fonction pour prédire les tags


def predict_tags(question_vector, threshold=0.01):
    predictions = model.predict(question_vector)
    tag_probabilities = [(top_tags[i], float(predictions[0][i]))
                         for i in range(len(top_tags))]
    filtered_tag_probabilities = [
        (tag, prob) for tag, prob in tag_probabilities if prob >= threshold]
    sorted_tag_probabilities = sorted(
        filtered_tag_probabilities, key=lambda x: x[1], reverse=True)
    top_5_tags = sorted_tag_probabilities[:10]  # Limite à 10 tags
    main_tag = top_5_tags[0][0] if top_5_tags else None
    return top_5_tags, main_tag

# Route POST pour suggérer des tags


@app.route('/suggestion', methods=['POST'])
def suggest_tags():
    try:
        if not request.is_json:
            return jsonify({'error': 'Invalid request format, JSON expected'}), 400

        question = request.json.get('question', '')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Transforme la question en vecteur BoW + SVD
        question_vector_bow_svd = transform_text_to_bow_svd(question)
        top_5_tags, main_tag = predict_tags(question_vector_bow_svd)

        response = {
            'main_tag': main_tag,
            'suggested_tags': [tag for tag, _ in top_5_tags]
        }
        return jsonify(response)

    except Exception as e:
        logging.error("Erreur lors de la suggestion de tags: " + e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
