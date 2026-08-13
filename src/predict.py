"""
Chargement du modele entraine et prediction du niveau de risque de
paludisme a partir de nouvelles observations (dict ou DataFrame).

Usage en ligne de commande :
    python src/predict.py --models-dir models --json '{"mois": 7, ...}'

Usage en tant que module (utilise par app/app.py) :
    from predict import RiskPredictor
    predictor = RiskPredictor("models")
    predictor.predict_one({...})
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


class RiskPredictor:
    """Encapsule le pipeline sklearn entraine + l'encodeur de labels."""

    def __init__(self, models_dir: str | Path = "models"):
        models_dir = Path(models_dir)
        self.model = joblib.load(models_dir / "best_model.joblib")
        self.label_encoder = joblib.load(models_dir / "label_encoder.joblib")
        self.feature_columns = joblib.load(models_dir / "feature_columns.joblib")

    def predict_one(self, observation: dict) -> dict:
        """Predit le niveau de risque pour une observation unique.

        `observation` doit contenir au minimum les colonnes utilisees a
        l'entrainement (voir models/feature_columns.joblib). Les colonnes
        manquantes sont laissees a NaN et gerees par l'imputer du pipeline.
        """
        row = {col: observation.get(col) for col in self.feature_columns}
        X = pd.DataFrame([row])
        pred_encoded = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        classes = self.label_encoder.classes_

        return {
            "niveau_risque_predit": self.label_encoder.inverse_transform([pred_encoded])[0],
            "probabilites": {cls: float(p) for cls, p in zip(classes, proba)},
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predit le niveau de risque pour un DataFrame de plusieurs lignes."""
        missing = [c for c in self.feature_columns if c not in df.columns]
        for col in missing:
            df[col] = None
        X = df[self.feature_columns]
        pred_encoded = self.model.predict(X)
        proba = self.model.predict_proba(X)
        out = df.copy()
        out["niveau_risque_predit"] = self.label_encoder.inverse_transform(pred_encoded)
        for i, cls in enumerate(self.label_encoder.classes_):
            out[f"proba_{cls}"] = proba[:, i]
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predit le niveau de risque de paludisme.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--json", required=True, help="Observation au format JSON")
    args = parser.parse_args()

    predictor = RiskPredictor(args.models_dir)
    observation = json.loads(args.json)
    result = predictor.predict_one(observation)
    print(json.dumps(result, ensure_ascii=False, indent=2))
