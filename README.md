# simplon_debug

Application Flask de classification d'images satellite.

Le projet charge un modèle Keras (`models/final_cnn.keras`), prétraite une image envoyée par l'utilisateur, retourne une prédiction parmi quatre classes (`desert`, `forest`, `meadow`, `mountain`) et permet d'enregistrer un feedback utilisateur dans MongoDB.

## Fonctionnalités

- Upload d'une image depuis l'interface web.
- Prédiction d'une classe satellite parmi 4 catégories.
- Affichage du score de confiance du modèle.
- Feedback loop utilisateur avec enregistrement en base MongoDB.
- Journalisation applicative avec rotation quotidienne des logs.
- Alertes SMTP sur erreurs critiques si les variables nécessaires sont renseignées.
- Intégration de `flask_monitoringdashboard` pour le suivi de l'application.
- Tests automatisés avec `pytest`.
- Exécution automatique de la CI sur `push` et `pull_request` vers `main`.

## Stack technique

- **Backend** : Flask
- **Modèle IA** : Keras avec backend Torch (`KERAS_BACKEND=torch`)
- **Traitement image** : Pillow, NumPy
- **Base de données** : MongoDB
- **Monitoring** : `flask_monitoringdashboard`
- **Tests** : Pytest
- **CI** : GitHub Actions

## Arborescence

```text
simplon_debug/
├── .github/
│   └── workflows/
│       └── main.yml
├── MongoDB/
│   └── init_db.py
├── models/
│   └── final_cnn.keras
├── templates/
│   ├── upload.html
│   ├── result.html
│   └── feedback_ok.html
├── tests/
│   └── test_app.py
├── app.py
└── requirements.txt
```

## Prérequis

- Python **3.12**
- Une instance **MongoDB** accessible si l'on veut activer la feedback loop
- Le fichier de modèle `models/final_cnn.keras`

## Installation

```bash
python -m venv .venv
```

### Sous Linux / macOS

```bash
source .venv/bin/activate
```

### Sous Windows

```bash
.venv\Scripts\activate
```

Puis installer les dépendances :

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Variables d'environnement

L'application charge un fichier `.env` au démarrage.

Exemple minimal :

```env
run_check=1
mongo_uri=mongodb://localhost:27017/
db_name=simplon_debug
collection_name=feedbacks
fromaddr=votre_adresse_mail@example.com
credentials=votre_mot_de_passe_ou_mot_de_passe_application
```

### Rôle des variables

- `run_check` : active les blocs d'initialisation dépendants de l'environnement, notamment MongoDB et le handler SMTP.
- `mongo_uri` : URI de connexion MongoDB.
- `db_name` : nom de la base utilisée pour les feedbacks.
- `collection_name` : nom de la collection MongoDB.
- `fromaddr` : adresse expéditrice utilisée pour les alertes SMTP.
- `credentials` : secret associé à l'adresse mail utilisée par le handler SMTP.

> Dans l'état actuel du code, si `run_check` n'est pas défini, l'initialisation MongoDB et SMTP est ignorée. L'application peut démarrer, mais l'enregistrement des feedbacks ne sera pas opérationnel.

## Initialisation de MongoDB

Un script est fourni pour préparer la base de données :

```bash
python MongoDB/init_db.py
```

Ce script :

- vérifie si la base existe déjà,
- propose sa suppression si elle est présente,
- recrée la base et la collection,
- insère puis supprime un document temporaire pour forcer l'initialisation.

## Lancer l'application

```bash
python app.py
```

Ensuite, ouvrir l'application dans un navigateur à l'adresse locale fournie par Flask, en général :

```text
http://127.0.0.1:5000/
```

## Parcours utilisateur

1. Ouvrir la page d'accueil.
2. Envoyer une image satellite depuis le formulaire.
3. L'application prétraite l'image, lance la prédiction et affiche :
   - la classe prédite,
   - le score de confiance,
   - les choix de feedback.
4. L'utilisateur peut corriger la prédiction via le formulaire de feedback.
5. Le feedback est enregistré dans MongoDB avec :
   - l'image encodée en base64,
   - la prédiction du modèle,
   - le score du modèle,
   - le label choisi par l'utilisateur,
   - la date de création.

## Journalisation et monitoring

### Logs

Le logger applicatif :

- écrit dans `logs/app.log`,
- applique une rotation quotidienne,
- conserve 14 jours d'historique,
- envoie les erreurs vers la sortie standard d'erreur,
- peut envoyer des alertes critiques par SMTP si la configuration est disponible.

### Monitoring

Le projet intègre `flask_monitoringdashboard` via un binding direct sur l'application Flask.

## Tests

Les tests automatisés sont regroupés dans `tests/test_app.py`.

Ils vérifient notamment :

- les doctests présents dans `app.py`,
- la disponibilité de la route `/`,
- le bon format de sortie du prétraitement d'image,
- la validation des extensions autorisées,
- la cohérence de la sortie du modèle (4 classes, probabilités valides, somme proche de 1).

Lancer les tests :

```bash
pytest tests/test_app.py -v
```

## Intégration continue

Le workflow GitHub Actions situé dans `.github/workflows/main.yml` :

- se déclenche sur `push` vers `main`,
- se déclenche aussi sur `pull_request` vers `main`,
- installe les dépendances,
- exécute les tests Pytest.

## Templates HTML

L'interface repose sur trois templates :

- `upload.html` : page d'envoi de l'image,
- `result.html` : affichage de la prédiction et collecte du feedback,
- `feedback_ok.html` : message de confirmation après soumission.

## Dépendances principales

Le fichier `requirements.txt` inclut notamment :

- Flask
- flask_monitoringdashboard
- keras
- torch
- numpy
- Pillow
- pymongo
- pytest
- python-dotenv