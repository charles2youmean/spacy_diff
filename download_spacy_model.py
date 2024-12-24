import spacy
import os

def download_model():
    try:
        spacy.cli.download("fr_core_news_sm")
        print("Modèle SpaCy téléchargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du téléchargement du modèle : {e}")

if __name__ == "__main__":
    download_model()
