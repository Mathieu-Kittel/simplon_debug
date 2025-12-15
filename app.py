import sys
import os

# ------------ BACKEND_KERAS ------------
os.environ["KERAS_BACKEND"] = "torch"

import logging
from logging.handlers import TimedRotatingFileHandler

import io
import base64

from flask import Flask, render_template, request, redirect, url_for
import flask_monitoringdashboard as dashboard
from werkzeug.utils import secure_filename

import numpy as np
import keras

from PIL import Image


# ---------------- Logger ---------------- #
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

# Au changement de fichier, on ajoute la date au .log en cours
app_handler.suffix = "%d_%m_%Y.log"

# On indique le format pour les logs (date, niveau, message)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
app_handler.setFormatter(formatter)
cmd_handler.setFormatter(formatter)

# Ajout des handlers à logger
logger.addHandler(app_handler)
logger.addHandler(cmd_handler)

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
    logger.error("Exception non gérée", exc_info=(exc_type, exc_value, exc_traceback))

# Changement de hook
sys.excepthook = handle_exception
logger.debug("Changement du hook.")

# ---------------- Config ----------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp"}
CLASSES       = ['desert', 'forest', 'meadow', 'mountain']

# Instance Flask
app = Flask(__name__)
logger.debug("Instance Flask OK")



# ---------------- Model ----------------+
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

    if "file" not in request.files:
        return redirect("/")
    
    file = request.files["file"]
    if file.filename == "" or not allowed_file(secure_filename(file.filename)):
        return redirect("/")

    raw = file.read()
    pil_img = Image.open(io.BytesIO(raw))
    img_array = preprocess_from_pil(pil_img, model_H_W)

    probs = model.predict(img_array, verbose=0)[0]
    cls_idx = int(np.argmax(probs))
    label = CLASSES[cls_idx]
    conf = float(probs[cls_idx])

    image_data_url = to_data_url(pil_img, fmt="JPEG")

    return render_template("result.html", 
                           image_data_url=image_data_url, 
                           predicted_label=label, 
                           confidence=conf, 
                           classes=CLASSES)

@app.route("/feedback", methods=["GET"])
def feedback_ok():
    """Affiche la page de confirmation de feedback (placeholder).

    Returns:
        Réponse HTML rendant le template "feedback_ok.html".
    """
    return render_template("feedback_ok.html")


# Ajout du dashboard Flask
dashboard.bind(app)
logger.debug("Dashboard flask OK")


if __name__ == "__main__":
    app.run(debug=False)
