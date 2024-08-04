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
