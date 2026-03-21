import sys
import os

# ------------ BACKEND_KERAS ------------
os.environ["KERAS_BACKEND"] = "torch"

import logging
from logging.handlers import (TimedRotatingFileHandler,
                              SMTPHandler) 

import io
import base64

from flask import Flask, render_template, request, redirect, url_for
import flask_monitoringdashboard as dashboard

from werkzeug.utils import secure_filename

import numpy as np
import keras

from PIL import Image

from pymongo        import MongoClient
from pymongo.errors import PyMongoError
from dotenv         import load_dotenv
from datetime       import datetime, timezone

load_dotenv()

# -------------- Run Check --------------- #
# run_check permet de skip une partie des imports .env
# pour éviter les erreurs pytest

if os.getenv("run_check"):
    run_check = 1
else:
    run_check = 0


# ---------------- Logger ---------------- #

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("app_log")
logger.setLevel(logging.DEBUG)

# On utilise TimeRotatingFileHandler pour créer un fichier log/jour
# On conserve 14 jours de logs
app_handler = TimedRotatingFileHandler(
    filename    = "logs/app.log",       
    when        = "midnight",          
    interval    = 1,               
    backupCount = 14,           
    encoding    = "utf-8"
)
app_handler.setLevel(logging.DEBUG)

# On ajoute un deuxième handler, niveau error, pour garder trace dans la console
cmd_handler = logging.StreamHandler()
cmd_handler.setLevel(logging.ERROR)

# Enfin un troisième handler, SMTP

if run_check: # Partie du code non exécutée lors des pytests.
    # Récupération des variables mail
    smtp_from = os.getenv("fromaddr")
    smtp_cred = os.getenv("credentials")

    # Handler smtp
    smtp_handler = SMTPHandler(
        mailhost=("smtp.gmail.com", 587),
        fromaddr=smtp_from,
        toaddrs=["exemple_mail@ggmail.com"],
        subject="Erreur Classificateur Satellite",
        credentials=(smtp_from, smtp_cred),
        secure=()
    )
    smtp_handler.setLevel(logging.CRITICAL)


# Au changement de fichier, on ajoute la date au .log en cours
app_handler.suffix = "%d_%m_%Y.log"

# On indique le format pour les logs (date, niveau, message)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
app_handler.setFormatter(formatter)
cmd_handler.setFormatter(formatter)
if run_check:
    smtp_handler.setFormatter(formatter)

# Ajout des handlers à logger
logger.addHandler(app_handler)
logger.addHandler(cmd_handler)
if run_check:
    logger.addHandler(smtp_handler)

# Log d'initialisation
logger.debug("Instanciation du logger.")


# -------------- excepthook --------------
# sys.excepthook définit le comportement en cas d'erreur inattendue.
# On modifie le comportement avec handle_exception

def handle_exception(exc_type, exc_value, exc_traceback):
    """En cas d'erreur inattendue, on passe par le logger.
    Ne prend pas en compte ctrl + c."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Exception non gérée", exc_info=(exc_type, exc_value, exc_traceback))

# Changement de hook
sys.excepthook = handle_exception
logger.debug("Changement du hook OK.")


# --------------- MongoDB --------------- #

if run_check:
    mongo_uri = os.getenv("mongo_uri")
    db_name = os.getenv("db_name")
    collection_name = os.getenv("collection_name")

    logger.debug("Chargement des variables MongoDB OK.")

    # Connexion au serveur/BDD/Collection Mongo
    mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    mongo_db = mongo_client[db_name]
    mongo_collection = mongo_db[collection_name]


    # Ping pour vérifier la connexion

    try:
        mongo_client.admin.command("ping")
        logger.info("Ping MongoDB OK.")
    except PyMongoError as e:
        logger.error("Connexion MongoDB échouée.", exc_info=e)
        raise



# ---------------- Config ----------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp"}
CLASSES       = ['desert', 'forest', 'meadow', 'mountain']

# Instance Flask
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="templates/css",   
    static_url_path="/css" 
)
logger.debug("Instance Flask OK")



# ---------------- Model ----------------
MODEL_PATH = "models/final_cnn.keras"
model = keras.saving.load_model(MODEL_PATH, compile=False)
logger.debug("Model correctement chargé.")

# On récupère les dimensions attendues par le model, pour traitement à venir
model_in_shape = model.input_shape
model_H_W = (model_in_shape[1], model_in_shape[2])
logger.debug(f"Input shape du model : {model_in_shape}")
logger.debug(f"Output shape du model : {model.output_shape}")

# ---------------- Utils ----------------
def allowed_file(filename: str) -> bool:
    """Vérifie si le nom de fichier possède une extension autorisée.
    La vérification est **insensible à la casse** et ne regarde que la sous-chaîne
    après le dernier point. Dépend de la constante globale `ALLOWED_EXT`.

    Args:
        filename: Nom du fichier soumis (ex. "photo.PNG").

    Returns:
        True si l’extension (ex. "png", "jpg") est dans `ALLOWED_EXT`, sinon False.

    Examples:
        >>> ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
        >>> allowed_file("img.JPG")
        True
        >>> allowed_file("archive.tar.gz")
        False
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def to_data_url(pil_img: Image.Image, fmt="JPEG") -> str:
    """Convertit une image PIL en Data URL base64 affichable dans un <img src="...">.
    L’image est encodée en mémoire (sans I/O disque), sérialisée en base64, puis
    encapsulée comme `data:<mime>;base64,<payload>`. Le type MIME est déduit de `fmt`.

    Args:
        pil_img: Image PIL à encoder.
        fmt: Format d’encodage PIL (ex. "JPEG", "PNG"). Par défaut "JPEG".

    Returns:
        Chaîne Data URL prête à être insérée dans une balise <img>.

    Raises:
        ValueError: si la sauvegarde PIL échoue pour le format demandé.

    Examples:
        >>> url = to_data_url(Image.new("RGB", (10, 10), "red"), fmt="PNG")
        >>> url.startswith("data:image/png;base64,")
        True
    """
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def preprocess_from_pil(pil_img: Image.Image, model_shape: tuple) -> np.ndarray:
    """Prépare une image PIL pour une prédiction Keras (redimentionnement, normalisation + batch).
    Convertit en RGB, normalise en [0, 1] (float32) et ajoute l’axe batch.

    Args:
        pil_img: Image PIL source.
        model_shape : dimensions attendues pour l'image, extraite du model

    Returns:
        np.ndarray de forme (1, H, W, 3), dtype float32, valeurs ∈ [0, 1].
    """
    # Conversion de l'image en RGB
    img = pil_img.convert("RGB")

    # On récupère la taille du plus grand côté de l'image, et de l'input shape
    img_max_side = max(img.size)
    model_max_side = max(model_shape)

    # On prévoit un Resampling avec LANCZOS, mais c'est lourd en calcul.
    # Donc si l'image est trop grande, on la redimentionne en deux fois.
    # On utilise alors BILUINEAR.
    if img_max_side > model_max_side * 7:
        img = img.resize((model_shape[0]*5, model_shape[1]*5), Image.Resampling.BILINEAR)

    # Passage à la taille attendue par le modèle.
    img = img.resize(model_shape, Image.Resampling.LANCZOS)

    # Normalisation
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

logger.info("Initialisation des variables et des fonctions\
réussie.")

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    """Affiche la page d’upload.

    Returns:
        Réponse HTML rendant le template "upload.html".
    """
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Traite l’upload, exécute la prédiction et affiche le résultat.

    Attendu: une requête `multipart/form-data` avec le champ `file`.
    Étapes:
      1) Validation de présence et d’extension du fichier.
      2) Lecture du contenu en mémoire et ouverture en PIL.
      3) Prétraitement -> tenseur (1, 224, 224, 3).
      4) Prédiction Keras -> probas, top-1 (label, confiance).
      5) Encodage de l’image en Data URL et rendu du template résultat.

    Redirects:
        - Redirige vers "/" si le fichier est manquant ou invalide.

    Returns:
        Réponse HTML rendant "result.html" avec:
        - `image_data_url` : image soumise encodée (base64),
        - `predicted_label` : classe prédite (str),
        - `confidence` : score softmax (float),
        - `classes` : liste des classes (pour les boutons).
    """

    # Vérification de la présence d'une image valide
    if "file" not in request.files:
        return redirect("/")
    
    file = request.files["file"]
    if file.filename == "" or not allowed_file(secure_filename(file.filename)):
        return redirect("/")

    # Lecture et traitement de l'image
    raw = file.read()
    pil_img = Image.open(io.BytesIO(raw))
    img_array = preprocess_from_pil(pil_img, model_H_W)

    # Prédiction
    probs = model.predict(img_array, verbose=0)[0]
    cls_idx = int(np.argmax(probs))
    label = CLASSES[cls_idx]
    conf = float(probs[cls_idx])

    # Transformation en url et base64
    image_data_url = to_data_url(pil_img, fmt="JPEG")

    return render_template("result.html", 
                           image_data_url=image_data_url, 
                           predicted_label=label, 
                           confidence=conf, 
                           classes=CLASSES)

@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Réception et stockage du feedback utilisateur en base MongoDB.
    Redirige vers la page de feedback.
    """

    # Message par défaut pour la page de feedback
    message = "Merci pour le feedback ! 👍"

    # Récupération des informations pour le feedback
    try:
        image_data_url   = request.form.get("image_base64")
        model_prediction = request.form.get("model_prediction")
        model_confidence = request.form.get("model_confidence")
        user_feedback    = request.form.get("user_feedback")

        # On retire le header HTML pour ne garder que la base64
        image_base64 = None
        if image_data_url and "," in image_data_url:
            image_base64 = image_data_url.split(",", 1)[1]

        # Constitution de l'objet à insérer en base de données
        doc = {
            "image_base64": image_base64,
            "model_prediction": model_prediction,
            "model_confidence": float(model_confidence) if model_confidence is not None else None,
            "user_feedback": user_feedback,
            "created_at": datetime.now(timezone.utc),
            "_init": True
        }

        # Insertion de l'objet dans MongoDB
        mongo_collection.insert_one(doc)
        logger.info("Feedback inséré en base MongoDB.")

    except Exception as e:
        logger.exception(f"Erreur lors de l'insertion du feedback :\n{e}")
        message = "Une erreur est survenue, merci d'avoir essayé. 😉"

    return render_template("feedback_ok.html", message=message)


# Ajout du dashboard Flask
dashboard.bind(app)
logger.debug("Dashboard flask OK")


if __name__ == "__main__":
    app.run(debug=False)
