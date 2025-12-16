# ---------- Imports ----------
import sys
import os

# On ajoute le dossier parent au pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from PIL import Image
from app import (app as flask_app,
                 model,
                 preprocess_from_pil as processing,
                 allowed_file)

# ---------- Variables ----------

# Image factice
# sert aux tests processing model shape et model_output
img = Image.new("RGB", (800, 800), color=(255, 0, 0))

# Hauteur/largeur d'entrée du model
# sert aux tests processing model shape et model_output
model_H_W = model.input_shape[1:3]


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client

def test_doctests():
    """Vérifie que tous les doctests dans app.py passent."""
    import doctest
    failures, tests = doctest.testmod(__import__("app"))
    assert failures == 0

def test_home_status(client):
    """La route '/' doit répondre avec le code HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200

def test_processing_model_shape():
    """Compare la sortie de la fonction de traitement d'image avec l'entrée du model.
    On compare les hauteurs et largeurs."""
    assert model_H_W == processing(img, model_H_W).shape[1:3]

def test_allowed():
    """Vérifie que le filtre d'extension fonctionne."""
    assert ( allowed_file("doc.pdf") == False 
         and allowed_file("test.png") == True)

def test_model_output():
    """Vérif que la sortie du model a la longueur attendue.
    La longueur doit être 4.
    Aucun élément < 0 ou > 1
    La somme des éléments == 1
    Note : on applique une tolérance pour la somme, liées aux imprécisions des floats.
    Tolérance : +/- 1e-6"""
    img_array = processing(img, model_H_W)
    probs = model.predict(img_array, verbose=0)[0]
    
    # La sortie doit avoir 4 éléments
    assert len(probs) == 4

    # Les probabilités doivent être comprises entre 0 et 1
    assert (probs >= 0).all()
    assert (probs <= 1).all()

    # La somme doit être (quasi) égale à 1
    assert abs(probs.sum() - 1.0) < 1e-6
    