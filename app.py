import pandas as pd
import spacy
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Charger le modèle SpaCy léger pour le français
nlp = spacy.load("fr_core_news_sm")

# Charger la liste des stop words depuis un fichier .txt
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stop_words = set(line.strip() for line in file)
    return stop_words

# Fonction pour analyser une colonne avec SpaCy, POS-tagging et stop words
def analyze_column_spacy(column, stop_words):
    lemmatized_words = []
    for text in column.dropna():  # Ignorer les cellules vides
        doc = nlp(str(text))
        lemmatized_words.extend(
            token.lemma_
            for token in doc
            if not token.is_punct                # Exclure la ponctuation
            and token.lemma_ not in stop_words   # Exclure les stop words
            and token.pos_ in {"NOUN", "ADJ", "VERB"}  # Garder seulement noms, adjectifs, verbes
        )
    total_lemmas = len(lemmatized_words)
    lemma_counts = pd.Series(lemmatized_words).value_counts()
    relative_freq = lemma_counts / total_lemmas
    return lemma_counts, relative_freq

# Fonction pour trouver les mots communs et différents
def compare_columns(freqs):
    common_words = pd.DataFrame(freqs).mean(axis=1).sort_values(ascending=False).head(50)
    diff_words = pd.DataFrame(freqs).std(axis=1).sort_values(ascending=False).head(50)
    return common_words, diff_words

# Fonction pour trouver les mots propres à chaque corpus
def unique_words_by_corpus(freqs, columns):
    singular_words = {}
    df_freqs = pd.DataFrame(freqs).fillna(0)  # Créer un DataFrame des fréquences
    for col in columns:
        # Calcul de la singularité : fréquence relative dans une colonne / somme des fréquences
        singularity = df_freqs[col] / df_freqs.sum(axis=1)
        singular_words[col] = singularity.sort_values(ascending=False).head(50)
    return singular_words

# Fonction pour générer un nuage de mots
def generate_wordcloud(word_freq, title):
    # Configurer le nuage de mots
    wordcloud = WordCloud(
        background_color="white",
        color_func=lambda *args, **kwargs: "blue",
        width=800,
        height=400,
        max_words=50
    ).generate_from_frequencies(word_freq)

    # Afficher le nuage de mots
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=12)
    st.pyplot(plt)

# Interface Streamlit
st.title("Analyse des verbatims avec SpaCy (POS-Tagging et stop words)")

uploaded_file = st.file_uploader("Téléchargez votre fichier Excel", type=["xlsx"])

if uploaded_file:
    data = pd.read_excel(uploaded_file)
    st.write("Aperçu du fichier chargé :")
    st.dataframe(data.head())

    # Sélectionner les colonnes
    columns = st.multiselect("Sélectionnez les colonnes à analyser", options=data.columns)

    if len(columns) >= 2:
        # Charger les stop words
        stop_words = load_stopwords("stopwords.txt")
        results = {}

        # Analyser chaque colonne
        for col in columns:
            lemma_counts, relative_freq = analyze_column_spacy(data[col], stop_words)
            results[col] = relative_freq

        # Comparer les colonnes
        common_words, diff_words = compare_columns(results)

        # Trouver les mots singuliers pour chaque corpus
        singular_words = unique_words_by_corpus(results, columns)

        # Afficher les résultats
        st.write("**Mots les plus communs (Top 50)**")
        st.write(common_words)
        generate_wordcloud(common_words, "Nuage des mots les plus communs")

        st.write("**Mots les plus différents (Top 50)**")
        st.write(diff_words)
        generate_wordcloud(diff_words, "Nuage des mots les plus différents")

        st.write("**Mots propres à chaque corpus (Top 50)**")
        for col, singularity in singular_words.items():
            st.write(f"Mots singuliers pour {col}:")
            st.write(singularity)
            generate_wordcloud(singularity, f"Nuage des mots singuliers pour {col}")
    else:
        st.warning("Veuillez sélectionner au moins 2 colonnes pour effectuer une comparaison.")
