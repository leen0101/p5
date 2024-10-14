# API de Traitement de Texte et Prédiction de Tags
Pour tester l'api: https://apip5-975919512217.us-central1.run.app/suggestion

## Introduction
Ce dépôt contient une API de traitement de texte et de prédiction de tags. L'API utilise le modèle BoW (Bag of Words) avec SVD de machine learning pour prédire les tags pertinents à partir d'une question fournie. Le projet est construit avec Flask pour l'API, tandis que les modèles sont entraînés et gérés via MLflow, en anglais. Les notebooks inclut plusieurs approches de traitement des données : Bag of Words (BoW) avec SVD, Word2Vec, et Universal Sentence Encoder (USE), avec des tests pour la partie de modèle non supervisée.

## Fonctionnalités
* Prétraitement du Texte : Nettoyage et préparation des données textuelles (suppression des balises HTML, ponctuation et mots vides).
* Prédiction de Tags : Prédiction des tags les plus probables associés à une question donnée en utilisant divers modèles de machine learning.
* Suivi avec MLflow : Tous les artefacts et métriques du modèle sont enregistrés et suivis dans MLflow.
* API Flask : Endpoint /suggestion qui permet de recevoir une question et retourne les tags suggérés.

### Modèles de Machine Learning
* Bag of Words (BoW) avec SVD : Transforme les textes en vecteurs avec BoW puis réduit la dimensionnalité avec SVD.
* Word2Vec : Entraîné sur les textes, transforme chaque document en vecteur par la moyenne des mots qui le composent.
* Universal Sentence Encoder (USE) : Utilise un modèle pré-entraîné pour capturer les relations sémantiques entre les phrases.

## Installation et Configuration
Prérequis
Python 3.12
pip pour gérer les dépendances
MLflow pour le suivi des modèles
TensorFlow 2.17 pour le modèle basé sur les réseaux de neurones
Gensim pour Word2Vec

## Installation
Clonez le dépôt :

```
git clone https://github.com/leen0101/p5.git
```

Installez les dépendances requises :

```
pip install -r requirements.txt
```

### Configurez les variables d'environnement pour accéder à Google Cloud et MLflow :

Ajoutez votre clé d'authentification Google Cloud à la variable GOOGLE_APPLICATION_CREDENTIALS.
Configurez l'URI de suivi MLflow via la variable MLFLOW_TRACKING_URI.

Lancez l'API :
```
python app.py
```

Utilisation de l'API
Endpoint /suggestion
Envoyez une requête POST avec une question pour obtenir des suggestions de tags.

Exemple de requête :

```
curl -X POST http://localhost:8080/suggestion -H "Content-Type: application/json" -d '{"question": "How to learn python?"}'
```
Exemple de réponse :
```
{
  "main_tag": "python",
  "suggested_tags": ["python", "learning", "programmation"]
}
```
Suivi des Modèles avec MLflow
Les modèles entraînés sont enregistrés dans MLflow pour suivre les métriques et les artefacts :

BoW + SVD : Sauvegardé sous forme de fichier .h5.
Word2Vec : Sauvegardé sous forme de modèle Gensim.
USE : Modèle TensorFlow sauvegardé avec MLflow.
Les artefacts des modèles (vecteurs, modèles) sont également enregistrés dans MLflow pour une réutilisation future.

Structure du Projet
notebooks/ : Contient les notebooks d'exploration, de nettoyage des données, et d'entraînement des modèles.
api/app.py : Code de l'API Flask qui sert les prédictions de tags.
mlflow_artifacts/ : Contient les artefacts des modèles (vecteurs, modèles).
db/questions_db.csv : Fichier de données d'entraînement.
Tests
Les tests sont exécutés via pytest. Vous pouvez exécuter les tests comme suit :

```
py test_app.py
```
Déploiement
L'API peut être déployée sur des services comme Google Cloud Run ou Heroku. 