"""
Fonctions de chargement et de prétraitement des données pour le projet
« Prédiction du risque de paludisme en Guinée ».

Reproduit fidèlement la logique du notebook d'analyse exploratoire :
- lecture robuste du CSV (encodage utf-8-sig)
- extraction d'une variable temporelle (jour de l'année)
- exclusion des colonnes provoquant une fuite de données
  (identifiant, date brute, année constante, variables sanitaires
  directement dérivées de la cible)
- séparation des variables numériques / catégorielles
- construction du ColumnTransformer (imputation + standardisation +
  encodage One-Hot)
"""

# Permet d'utiliser la syntaxe moderne des annotations de type (ex: "str | Path")
# meme sur des versions de Python anterieures a 3.10.
from __future__ import annotations

# Classe pratique pour manipuler des chemins de fichiers de facon portable.
from pathlib import Path

# Bibliotheque de calcul numerique (utilisee ici surtout pour detecter les types numeriques).
import numpy as np

# Bibliotheque de manipulation de donnees tabulaires (DataFrame).
import pandas as pd

# ColumnTransformer permet d'appliquer des transformations differentes selon les colonnes
# (ex: une pipeline pour les colonnes numeriques, une autre pour les colonnes categorielles).
from sklearn.compose import ColumnTransformer

# SimpleImputer permet de remplacer les valeurs manquantes (NaN) par une valeur calculee
# (ex: la mediane pour le numerique, la valeur la plus frequente pour le categoriel).
from sklearn.impute import SimpleImputer

# Pipeline permet d'enchainer plusieurs etapes de transformation (ex: imputation puis mise a l'echelle)
# comme une seule etape coherente.
from sklearn.pipeline import Pipeline

# LabelEncoder : transforme des labels textuels (ex: "faible", "moyen", "eleve") en entiers.
# OneHotEncoder : transforme une variable categorielle en plusieurs colonnes binaires (0/1).
# StandardScaler : centre et reduit les variables numeriques (moyenne 0, ecart-type 1).
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Nom de la colonne cible (la variable que le modele doit predire) : le niveau de risque.
TARGET = "niveau_risque"

# Colonnes exclues des variables explicatives :
# - id_observation : identifiant, sans valeur prédictive
# - date / annee   : redondantes avec jour_annee / mois / semaine
# - cas_paludisme_simules, taux_incidence_simule_pour_1000 : dérivent
#   directement la cible -> fuite de données si conservées.
# On definit ici un "set" (ensemble) Python regroupant les noms de colonnes
# a ne jamais utiliser comme variable explicative (feature) pour l'entrainement.
EXCLUDED_COLUMNS = {
    "id_observation",              # identifiant unique de chaque ligne, aucune information predictive
    "date",                        # date brute au format texte, remplacee par jour_annee (numerique)
    "annee",                       # l'annee est jugee constante/peu utile ici, ecartee
    TARGET,                        # on exclut la cible elle-meme des variables explicatives (evite les fuites triviales)
    "cas_paludisme_simules",              # variable directement liee/calculee a partir de la cible -> fuite de donnees
    "taux_incidence_simule_pour_1000",    # idem : derive directement du niveau de risque -> fuite de donnees
}


# Fonction qui charge le fichier CSV brut en DataFrame pandas.
# "path" peut etre une chaine de caracteres ou un objet Path (chemin vers le fichier).
def load_data(path: str | Path) -> pd.DataFrame:
    """Charge le CSV brut avec un encodage tolérant aux exports Excel."""
    # Lit le fichier CSV avec l'encodage "utf-8-sig", qui gere correctement le BOM
    # (byte order mark) souvent ajoute par Excel lors de l'export en CSV,
    # evitant ainsi des caracteres bizarres au debut du fichier (ex: premiere colonne mal nommee).
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Retourne le DataFrame charge tel quel, sans autre transformation.
    return df


# Fonction qui ajoute une nouvelle variable temporelle derivee de la colonne "date".
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la variable jour_annee à partir de la colonne date."""
    # Fait une copie du DataFrame en entree pour eviter de modifier l'original
    # passe en argument (bonne pratique pour eviter les effets de bord).
    df = df.copy()

    # Convertit la colonne "date" (probablement du texte) en veritables objets datetime pandas.
    # errors="coerce" signifie que toute valeur qui ne peut pas etre convertie en date
    # sera remplacee par NaT (Not a Time, l'equivalent de NaN pour les dates),
    # plutot que de faire planter le programme.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Extrait le "jour de l'annee" (un nombre entre 1 et 365/366) a partir de la date,
    # et le stocke dans une nouvelle colonne "jour_annee". C'est une facon de representer
    # la saisonnalite de maniere numerique, utilisable par le modele.
    df["jour_annee"] = df["date"].dt.dayofyear

    # Retourne le DataFrame enrichi de cette nouvelle colonne.
    return df


# Fonction principale qui prepare les donnees pour l'entrainement :
# separe les variables explicatives (X) de la variable cible encodee (y),
# et identifie quelles colonnes sont numeriques et lesquelles sont categorielles.
def split_features_target(df: pd.DataFrame):
    """Sépare les variables explicatives (X) de la cible encodée (y).

    Retourne également l'encodeur de labels (pour retrouver les noms de
    classes) et les listes de colonnes numériques / catégorielles.
    """
    # Ajoute d'abord la colonne "jour_annee" (voir add_temporal_features ci-dessus),
    # necessaire pour disposer d'une information temporelle exploitable par le modele.
    df_model = add_temporal_features(df)

    # Construit la liste des colonnes a utiliser comme variables explicatives :
    # on garde toutes les colonnes du DataFrame SAUF celles listees dans EXCLUDED_COLUMNS.
    feature_columns = [c for c in df_model.columns if c not in EXCLUDED_COLUMNS]

    # Cree le DataFrame X qui ne contient que les colonnes explicatives selectionnees.
    # .copy() evite de modifier accidentellement df_model par la suite.
    X = df_model[feature_columns].copy()

    # Instancie un encodeur de labels, qui va convertir les valeurs textuelles
    # de la colonne cible (ex: "faible", "moyen", "eleve") en entiers (ex: 0, 1, 2).
    label_encoder = LabelEncoder()

    # Applique l'encodeur sur la colonne cible convertie en chaines de caracteres (astype(str),
    # utile si jamais la colonne contenait des types mixtes), et recupere les valeurs encodees (y).
    # fit_transform apprend les classes possibles ET transforme les donnees en une seule etape.
    y = label_encoder.fit_transform(df_model[TARGET].astype(str))

    # Identifie automatiquement les colonnes de type "object" ou "category" dans X,
    # c'est-a-dire les colonnes contenant du texte/categories (variables categorielles).
    # .tolist() convertit l'index pandas resultant en simple liste Python.
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Identifie automatiquement les colonnes de type numerique (int, float, etc.) dans X,
    # grace a np.number qui englobe tous les types numeriques de numpy.
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Retourne un tuple contenant : les variables explicatives (X), la cible encodee (y),
    # l'encodeur de labels (utile plus tard pour retrouver les noms textuels des classes),
    # et les deux listes de colonnes (numeriques et categorielles).
    return X, y, label_encoder, numeric_features, categorical_features


# Fonction qui construit le "preprocesseur" : un objet sklearn qui applique
# automatiquement les bonnes transformations a chaque type de colonne (numerique/categorielle).
def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Construit le ColumnTransformer : imputation + normalisation + One-Hot."""
    # On essaie d'abord de creer un OneHotEncoder avec l'argument "sparse_output=False",
    # qui est le nom utilise dans les versions recentes de scikit-learn
    # (il force l'encodeur a produire un tableau numpy classique plutot qu'une matrice creuse).
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    # Si cet argument n'existe pas (ancienne version de scikit-learn ou l'argument s'appelait
    # "sparse" au lieu de "sparse_output"), Python leve une TypeError : on rattrape cette erreur
    # et on utilise l'ancien nom d'argument a la place, pour rester compatible.
    except TypeError:  # anciennes versions de scikit-learn
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    # Dans les deux cas, handle_unknown="ignore" signifie que si une categorie inconnue
    # (jamais vue a l'entrainement) apparait plus tard lors d'une prediction, l'encodeur
    # ne plantera pas mais mettra simplement des zeros partout pour cette observation.

    # Cree le ColumnTransformer, qui va appliquer differentes transformations
    # selon qu'une colonne est numerique ou categorielle.
    preprocessor = ColumnTransformer(
        transformers=[
            # Premier bloc de transformation : nomme "num", applique aux colonnes numeriques.
            (
                "num",
                # Pipeline appliquee uniquement aux colonnes numeriques :
                Pipeline(
                    [
                        # Etape 1 : remplace les valeurs manquantes par la mediane de la colonne
                        # (la mediane est robuste aux valeurs extremes, contrairement a la moyenne).
                        ("imputer", SimpleImputer(strategy="median")),
                        # Etape 2 : standardise les donnees (moyenne = 0, ecart-type = 1),
                        # ce qui aide de nombreux algorithmes de machine learning a mieux converger/performer.
                        ("scaler", StandardScaler()),
                    ]
                ),
                # Liste des colonnes auxquelles cette pipeline numerique doit s'appliquer.
                numeric_features,
            ),
            # Deuxieme bloc de transformation : nomme "cat", applique aux colonnes categorielles.
            (
                "cat",
                # Pipeline appliquee uniquement aux colonnes categorielles :
                Pipeline(
                    [
                        # Etape 1 : remplace les valeurs manquantes par la valeur la plus frequente
                        # de la colonne (logique pour des donnees categorielles, ou la mediane n'a pas de sens).
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        # Etape 2 : applique l'encodage One-Hot, qui transforme chaque categorie
                        # en une colonne binaire distincte (0 ou 1), format comprehensible par les modeles.
                        ("onehot", one_hot),
                    ]
                ),
                # Liste des colonnes auxquelles cette pipeline categorielle doit s'appliquer.
                categorical_features,
            ),
        ],
        # Indique que toute colonne non explicitement listee (ni numerique ni categorielle geree)
        # doit etre simplement supprimee du resultat final, plutot que conservee telle quelle.
        remainder="drop",
    )
    # Retourne l'objet ColumnTransformer configure, pret a etre utilise dans un pipeline global
    # (typiquement combine avec un modele de classification).
    return preprocessor


# Fonction utilitaire qui sauvegarde une version "nettoyee" du dataset (avec la colonne jour_annee ajoutee)
# dans un fichier CSV, par exemple pour inspection manuelle ou reutilisation ulterieure.
def save_processed_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    """Sauvegarde une version nettoyée (avec jour_annee) dans data/processed/."""
    # Convertit output_path (chaine ou Path) en objet Path pour pouvoir manipuler
    # facilement le dossier parent et creer les dossiers manquants si besoin.
    output_path = Path(output_path)

    # Cree le(s) dossier(s) parent(s) du fichier de sortie s'ils n'existent pas encore.
    # parents=True permet de creer toute l'arborescence de dossiers necessaire (pas seulement le dernier niveau).
    # exist_ok=True evite une erreur si le dossier existe deja.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Applique la fonction add_temporal_features pour obtenir la version enrichie
    # du DataFrame (avec la colonne jour_annee), sans modifier l'original.
    processed = add_temporal_features(df)

    # Ecrit le DataFrame traite dans un fichier CSV a l'emplacement indique.
    # index=False evite d'ecrire l'index pandas (0, 1, 2, ...) comme une colonne supplementaire dans le fichier.
    processed.to_csv(output_path, index=False)
