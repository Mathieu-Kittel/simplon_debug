import os
from datetime import datetime, timezone
from pymongo  import MongoClient
from dotenv   import load_dotenv

# Chargement .env
# -------------------
load_dotenv()

mongo_uri = os.getenv("mongo_uri")
db_name = os.getenv("db_name")
collection_name = os.getenv("collection_name")


def confirm(prompt: str) -> bool:
    """
    Demande une confirmation à l'utilisateur.
    Retourne True si la réponse est 'y'.

    Params:
    prompt: str - Le texte affiché à l'utilisateur
    """
    reply = input(f"{prompt} [y/N]: ").strip().lower()
    return reply == "y"

def main():
    """
    Script d'initialisation de la base MongoDB pour la feedback loop.
    - Vérifie l'existence de la base
    - Propose suppression si elle existe
    - Crée la base et la collection
    """

    # Connexion MongoDB
    client = MongoClient(mongo_uri)

    # On liste donc les bases MongoDB existantes
    # Si la notre existe déjà, on prévient que l'opération va la supprimer.
    existing_dbs = client.list_database_names()

    if db_name in existing_dbs:
        print(f"⚠️ La base '{db_name}' existe déjà.")
        if not confirm("La supprimer et la recréer ?"):
            print("Opération annulée.")
            return

        # Suppression de la base
        client.drop_database(db_name)
        print("Base supprimée.")

    # Accès à la base et à la collection
    db = client[db_name]
    collection = db[collection_name]

    # -----------------------------------------------------------------
    # La base MongoDB est créé à la première insertion.
    # On insère un objet fantôme pour "créer" la base
    # Puis on supprimer l'objet.
    collection.insert_one({
        "image_base64": None,
        "model_prediction": None,
        "model_score": None,
        "user_feedback": None,
        "created_at": datetime.now(timezone.utc),
        "_init": True  # marqueur interne temporaire
    })

    # Suppression immédiate du document d'initialisation
    collection.delete_one({"_init": True})

    print(f"✅ Base '{db_name}' et collection '{collection_name}' prêtes.")


if __name__ == "__main__":
    main()