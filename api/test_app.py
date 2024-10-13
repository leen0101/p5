import pytest
from flask import Flask
from app import app
import json

# Fixture pour configurer le client de test Flask


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test: Requête valide avec unne question


def test_valid_request(client):
    payload = {'question': 'How to learn Python programming?'}
    response = client.post('/suggestion', json=payload)
    data = json.loads(response.data)

    assert response.status_code == 200
    assert 'main_tag' in data
    assert 'suggested_tags' in data
    assert isinstance(data['suggested_tags'], list)

# Test: Requête sans le paramètre 'question'


def test_missing_question(client):
    payload = {}
    response = client.post('/suggestion', json=payload)
    data = json.loads(response.data)

    assert response.status_code == 400
    assert 'error' in data
    assert data['error'] == 'No question provided'

# Test: Requête avec un format non-JSON


def test_invalid_format(client):
    payload = "This is not JSON"
    response = client.post('/suggestion', data=payload)

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Invalid request format, JSON expected'

# Test: Requête avec simulation d'une erreur dans la fonction `transform_text_to_bow_svd`


def test_internal_server_error(client, mocker):
    mocker.patch('app.transform_text_to_bow_svd',
                 side_effect=Exception('Test error'))
    payload = {'question': 'How to learn Python programming?'}
    response = client.post('/suggestion', json=payload)

    assert response.status_code == 500

    try:
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Test error'
    except json.JSONDecodeError:
        print(f"Réponse non JSON : {response.data}")
        raise
