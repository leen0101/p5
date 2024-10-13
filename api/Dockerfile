# Image TensorFlow
FROM tensorflow/tensorflow:2.17.0

# Définit le répertoire de travail
WORKDIR /app

# Copie tous les fichiers dans le conteneur
COPY . /app


# Installe les dépendances
RUN pip install --upgrade pip
RUN pip install --ignore-installed -r requirements.txt


# Expose le port pour Flask
EXPOSE 8080

# Commande pour démarrer l'application avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

