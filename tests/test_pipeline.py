"""Tests de base : chargement des données, prétraitement, inference."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from preprocessing import load_data, split_features_target  # noqa: E402

DATA_PATH = ROOT_DIR / "data" / "raw" / "paludisme.csv"
MODELS_DIR = ROOT_DIR / "models"


def test_load_data():
    df = load_data(DATA_PATH)
    assert len(df) > 0
    assert "niveau_risque" in df.columns


def test_split_features_target_no_leakage():
    df = load_data(DATA_PATH)
    X, y, label_encoder, numeric_features, categorical_features = split_features_target(df)
    leaky_columns = {"cas_paludisme_simules", "taux_incidence_simule_pour_1000", "id_observation", "date", "annee"}
    assert leaky_columns.isdisjoint(set(X.columns))
    assert len(y) == len(df)
    assert set(label_encoder.classes_) == {"faible", "moyen", "eleve"}


@pytest.mark.skipif(not (MODELS_DIR / "best_model.joblib").exists(), reason="Modele non entraine")
def test_predict_one():
    from predict import RiskPredictor

    predictor = RiskPredictor(MODELS_DIR)
    observation = {
        "mois": 8, "semaine": 32, "region_administrative": "Nzérékoré",
        "prefecture": "Nzérékoré", "zone_climatique": "forestier",
        "latitude": 7.75, "longitude": -8.82,
        "temperature_moyenne_c": 26.5, "temperature_min_c": 21.0, "temperature_max_c": 31.0,
        "precipitation_mm": 15.0, "precipitation_7j_mm": 90.0, "precipitation_30j_mm": 250.0,
        "humidite_relative_pct": 82.0, "vitesse_vent_kmh": 8.0, "jours_pluie_7j": 6,
        "humidite_sol_pct": 70.0, "indice_eau_stagnante": 0.8, "jour_annee": 220,
    }
    result = predictor.predict_one(observation)
    assert result["niveau_risque_predit"] in {"faible", "moyen", "eleve"}
    assert abs(sum(result["probabilites"].values()) - 1.0) < 1e-6
