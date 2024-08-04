from flask import Flask, request, jsonify
import tensorflow as tf
import pickle

app = Flask(__name__)


def load_data(x_path='../notebooks/model/X_reduced.pkl', y_path='../notebooks/model/y.pkl'):
    with open(x_path, 'rb') as f:
        X = pickle.load(f)
    with open(y_path, 'rb') as f:
        y = pickle.load(f)
    return X, y


def load_model_objects():
    with open('../notebooks/model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('../notebooks/model/svd.pkl', 'rb') as f:
        svd = pickle.load(f)
    with open('../notebooks/model/top_tags.pkl', 'rb') as f:
        top_tags = pickle.load(f)
    return vectorizer, svd, top_tags


X, y = load_data()
vectorizer, svd, top_tags = load_model_objects()

model = tf.keras.models.load_model('../notebooks/best_model.keras')


def transform_text_to_bow(text):
    X_bow = vectorizer.transform([text])
    return svd.transform(X_bow)


@app.route('/predict_tags', methods=['POST'])
def predict_tags():
    question_text = request.json.get('question_text')
    if not question_text:
        return jsonify({'error': 'No question text provided'}), 400

    question_vector = transform_text_to_bow(question_text)
    predicted_tags_probabilities = model.predict(question_vector)
    predicted_tags = (predicted_tags_probabilities > 0.1).astype(int)
    predicted_tag_names = [top_tags[i] for i in range(
        len(predicted_tags[0])) if predicted_tags[0][i] == 1]

    return jsonify({'predicted_tags': predicted_tag_names, 'probabilities': predicted_tags_probabilities.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
