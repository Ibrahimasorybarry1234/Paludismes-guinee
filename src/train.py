"""
Entraînement et comparaison de trois modèles de classification pour
prédire le niveau de risque de paludisme (faible / moyen / eleve) en
Guinée à partir de variables météorologiques et environnementales.

Usage :
    python src/train.py --data data/raw/paludisme.csv --outdir models

Produit :
    models/best_model.joblib   -> pipeline complet (prétraitement + modèle)
    models/label_encoder.joblib
    models/feature_columns.joblib
    models/metrics.json        -> comparaison des 3 modèles
    data/processed/paludisme_clean.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from preprocessing import (
    build_preprocessor,
    load_data,
    save_processed_dataset,
    split_features_target,
)

RANDOM_STATE = 42


def build_models(preprocessor, n_classes: int) -> dict[str, Pipeline]:
    models = {
        "Regression_logistique": Pipeline(
            [
                ("preprocessing", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random_Forest": Pipeline(
            [
                ("preprocessing", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline(
            [
                ("preprocessing", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        num_class=n_classes,
                        random_state=RANDOM_STATE,
                        n_jobs=2,
                    ),
                ),
            ]
        )
    else:
        print("[avertissement] xgboost indisponible dans cet environnement : "
              "installez-le (voir requirements.txt) pour inclure le 3e modele.")
    return models


def main(data_path: str, outdir: str) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    save_processed_dataset(df, "data/processed/paludisme_clean.csv")

    X, y, label_encoder, numeric_features, categorical_features = split_features_target(df)
    class_names = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor, n_classes=len(class_names))

    train_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    results = []
    fitted_models = {}
    for name, model in models.items():
        print(f"Entrainement : {name}")
        if name == "XGBoost":
            model.fit(X_train, y_train, model__sample_weight=train_sample_weight)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        fitted_models[name] = model

        report = classification_report(
            y_test, y_pred, labels=np.arange(len(class_names)),
            target_names=class_names, zero_division=0, output_dict=True,
        )
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

        results.append(
            {
                "modele": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "rappel_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "roc_auc_ovr_weighted": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"),
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
            }
        )

    results_df = pd.DataFrame(results).set_index("modele").sort_values("f1_weighted", ascending=False)
    best_model_name = results_df.index[0]
    best_model = fitted_models[best_model_name]

    print("\nComparaison des modeles (tries par F1 pondere) :")
    print(results_df[["accuracy", "precision_weighted", "rappel_weighted", "f1_weighted", "roc_auc_ovr_weighted"]])
    print(f"\nMeilleur modele retenu : {best_model_name}")

    joblib.dump(best_model, outdir_path / "best_model.joblib")
    joblib.dump(label_encoder, outdir_path / "label_encoder.joblib")
    joblib.dump(list(X.columns), outdir_path / "feature_columns.joblib")
    joblib.dump(
        {"numeric_features": numeric_features, "categorical_features": categorical_features},
        outdir_path / "feature_types.joblib",
    )

    metrics_export = {"best_model": best_model_name, "comparaison": {}}
    metrics_export["comparaison"] = {
        name: {
            "accuracy": float(results_df.loc[name, "accuracy"]),
            "precision_weighted": float(results_df.loc[name, "precision_weighted"]),
            "rappel_weighted": float(results_df.loc[name, "rappel_weighted"]),
            "f1_weighted": float(results_df.loc[name, "f1_weighted"]),
            "roc_auc_ovr_weighted": float(results_df.loc[name, "roc_auc_ovr_weighted"]),
        }
        for name in results_df.index
    }
    metrics_export["class_names"] = class_names

    with open(outdir_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, ensure_ascii=False, indent=2)

    print(f"\nModele sauvegarde dans {outdir_path}/best_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraine les modeles de prediction du risque de paludisme.")
    parser.add_argument("--data", default="data/raw/paludisme.csv", help="Chemin du CSV brut")
    parser.add_argument("--outdir", default="models", help="Dossier de sortie pour les modeles")
    args = parser.parse_args()
    main(args.data, args.outdir)
