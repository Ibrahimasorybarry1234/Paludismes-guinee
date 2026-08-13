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
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

TARGET = "niveau_risque"

# Colonnes exclues des variables explicatives :
# - id_observation : identifiant, sans valeur prédictive
# - date / annee   : redondantes avec jour_annee / mois / semaine
# - cas_paludisme_simules, taux_incidence_simule_pour_1000 : dérivent
#   directement la cible -> fuite de données si conservées.
EXCLUDED_COLUMNS = {
    "id_observation",
    "date",
    "annee",
    TARGET,
    "cas_paludisme_simules",
    "taux_incidence_simule_pour_1000",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Charge le CSV brut avec un encodage tolérant aux exports Excel."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la variable jour_annee à partir de la colonne date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["jour_annee"] = df["date"].dt.dayofyear
    return df


def split_features_target(df: pd.DataFrame):
    """Sépare les variables explicatives (X) de la cible encodée (y).

    Retourne également l'encodeur de labels (pour retrouver les noms de
    classes) et les listes de colonnes numériques / catégorielles.
    """
    df_model = add_temporal_features(df)

    feature_columns = [c for c in df_model.columns if c not in EXCLUDED_COLUMNS]
    X = df_model[feature_columns].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_model[TARGET].astype(str))

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    return X, y, label_encoder, numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Construit le ColumnTransformer : imputation + normalisation + One-Hot."""
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # anciennes versions de scikit-learn
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def save_processed_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    """Sauvegarde une version nettoyée (avec jour_annee) dans data/processed/."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = add_temporal_features(df)
    processed.to_csv(output_path, index=False)
