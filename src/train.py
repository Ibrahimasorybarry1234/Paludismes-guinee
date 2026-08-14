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

# Permet d'utiliser la syntaxe moderne des annotations de type (ex: "dict[str, Pipeline]")
# meme sur des versions de Python anterieures a 3.10.
from __future__ import annotations

# Module standard pour gerer les arguments passes en ligne de commande (--data, --outdir).
import argparse

# Module standard pour lire/ecrire des donnees au format JSON (ici : le fichier metrics.json).
import json

# Classe pratique pour manipuler des chemins de fichiers de facon portable.
from pathlib import Path

# Bibliotheque utilisee pour sauvegarder (serialiser) sur disque les objets Python
# entraines : le modele, l'encodeur de labels, la liste des colonnes, etc.
import joblib

# Bibliotheque de calcul numerique, utilisee ici notamment pour generer des tableaux d'indices.
import numpy as np

# Bibliotheque de manipulation de donnees tabulaires (DataFrame), utilisee pour
# organiser et trier les resultats de comparaison des modeles.
import pandas as pd

# Modele d'ensemble base sur des arbres de decision (foret aleatoire).
from sklearn.ensemble import RandomForestClassifier

# Modele lineaire de classification (regression logistique).
from sklearn.linear_model import LogisticRegression

# Fonctions de scikit-learn pour evaluer la qualite des predictions d'un modele :
# accuracy_score       -> proportion de predictions correctes
# classification_report-> tableau detaille precision/rappel/f1 par classe
# confusion_matrix      -> tableau croise predictions vs realite
# f1_score              -> moyenne harmonique precision/rappel
# precision_score       -> proportion de vrais positifs parmi les positifs predits
# recall_score          -> proportion de vrais positifs parmi les positifs reels
# roc_auc_score         -> aire sous la courbe ROC (qualite de separation des classes)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Fonction qui divise un jeu de donnees en un ensemble d'entrainement et un ensemble de test.
from sklearn.model_selection import train_test_split

# Pipeline permet d'enchainer le pretraitement et le modele comme une seule etape coherente,
# garantissant que le meme pretraitement est applique a l'entrainement et a la prediction.
from sklearn.pipeline import Pipeline

# Fonction qui calcule un poids pour chaque observation d'entrainement, de facon a
# compenser un dataset desequilibre (certaines classes moins representees que d'autres).
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost est une bibliotheque tierce (pas installee par defaut avec scikit-learn),
# on essaie donc de l'importer, et si elle n'est pas disponible on continue sans planter.
try:
    # Modele de boosting de gradient, souvent tres performant pour la classification.
    from xgboost import XGBClassifier

    # Drapeau qui indique que XGBoost est bien disponible dans l'environnement.
    XGBOOST_AVAILABLE = True
# Si le module xgboost n'est pas installe, Python leve une ImportError qu'on capture ici.
except ImportError:
    # On desactive simplement l'utilisation de XGBoost plutot que de faire planter le script.
    XGBOOST_AVAILABLE = False

# Importe les fonctions definies dans le fichier preprocessing.py du meme projet :
# build_preprocessor    -> construit le ColumnTransformer (imputation + scaling + one-hot)
# load_data             -> charge le CSV brut
# save_processed_dataset-> sauvegarde une version nettoyee du dataset
# split_features_target -> separe X (features) et y (cible encodee)
from preprocessing import (
    build_preprocessor,
    load_data,
    save_processed_dataset,
    split_features_target,
)

# Graine aleatoire fixe, utilisee partout ou le hasard intervient (split train/test, modeles),
# afin que les resultats soient reproductibles d'une execution a l'autre.
RANDOM_STATE = 42


# Fonction qui construit un dictionnaire de pipelines (pretraitement + modele) a comparer.
# "preprocessor" est le ColumnTransformer partage par tous les modeles.
# "n_classes" est le nombre de classes cibles (ex: 3 pour faible/moyen/eleve), necessaire a XGBoost.
def build_models(preprocessor, n_classes: int) -> dict[str, Pipeline]:
    # Dictionnaire associant un nom de modele (cle) a son pipeline complet (valeur).
    models = {
        # Premier modele : regression logistique.
        "Regression_logistique": Pipeline(
            [
                # Etape 1 du pipeline : applique le pretraitement (imputation, scaling, one-hot)
                # partage entre tous les modeles.
                ("preprocessing", preprocessor),
                # Etape 2 du pipeline : le modele de regression logistique lui-meme.
                # max_iter=2000 augmente le nombre maximal d'iterations pour l'algorithme
                # d'optimisation (pour eviter les erreurs de non-convergence).
                # class_weight="balanced" ajuste automatiquement les poids des classes
                # en fonction de leur frequence, pour compenser un dataset desequilibre.
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        # Deuxieme modele : foret aleatoire (Random Forest).
        "Random_Forest": Pipeline(
            [
                # Etape 1 : meme pretraitement partage.
                ("preprocessing", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,        # nombre d'arbres de decision dans la foret
                        max_depth=12,             # profondeur maximale de chaque arbre (limite le surapprentissage)
                        min_samples_leaf=2,       # nombre minimal d'observations requis dans une feuille
                        class_weight="balanced_subsample",  # rééquilibre les classes a chaque sous-echantillon (bootstrap)
                        random_state=RANDOM_STATE,           # graine aleatoire pour la reproductibilite
                        n_jobs=-1,                            # utilise tous les coeurs CPU disponibles pour accelerer l'entrainement
                    ),
                ),
            ]
        ),
    }
    # N'ajoute le modele XGBoost que si la bibliotheque a bien pu etre importee plus haut.
    if XGBOOST_AVAILABLE:
        # Troisieme modele (optionnel) : XGBoost.
        models["XGBoost"] = Pipeline(
            [
                # Etape 1 : meme pretraitement partage que les autres modeles.
                ("preprocessing", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,               # nombre d'arbres successifs (iterations de boosting)
                        max_depth=5,                     # profondeur maximale de chaque arbre
                        learning_rate=0.05,              # taux d'apprentissage : plus petit = apprentissage plus prudent/lent
                        subsample=0.85,                  # fraction des observations utilisees a chaque iteration (regularisation)
                        colsample_bytree=0.85,           # fraction des colonnes/features utilisees a chaque arbre (regularisation)
                        objective="multi:softprob",      # fonction objectif adaptee a la classification multi-classes avec probabilites
                        eval_metric="mlogloss",          # metrique d'evaluation interne : perte logarithmique multi-classes
                        num_class=n_classes,             # nombre de classes cibles a predire
                        random_state=RANDOM_STATE,       # graine aleatoire pour la reproductibilite
                        n_jobs=2,                         # nombre de threads CPU utilises par XGBoost
                    ),
                ),
            ]
        )
    # Si XGBoost n'est pas disponible, on informe simplement l'utilisateur via un message,
    # sans interrompre le script (les deux autres modeles seront quand meme entraines).
    else:
        print("[avertissement] xgboost indisponible dans cet environnement : "
              "installez-le (voir requirements.txt) pour inclure le 3e modele.")
    # Retourne le dictionnaire complet des modeles (2 ou 3 selon la disponibilite de XGBoost).
    return models


# Fonction principale qui orchestre tout le processus d'entrainement :
# chargement des donnees, entrainement des modeles, evaluation, sauvegarde des resultats.
# "data_path" est le chemin vers le CSV brut, "outdir" le dossier ou sauvegarder les modeles/metriques.
def main(data_path: str, outdir: str) -> None:
    # Convertit outdir (chaine) en objet Path pour faciliter la manipulation de chemins.
    outdir_path = Path(outdir)
    # Cree le dossier de sortie (et ses parents) s'il n'existe pas deja, sans erreur si deja present.
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Charge le fichier CSV brut en DataFrame pandas via la fonction du module preprocessing.
    df = load_data(data_path)
    # Sauvegarde une version nettoyee/enrichie (avec la colonne jour_annee) dans un fichier
    # dedie, pour inspection ou reutilisation ulterieure (independamment de l'entrainement).
    save_processed_dataset(df, "data/processed/paludisme_clean.csv")

    # Separe les variables explicatives (X), la cible encodee (y), l'encodeur de labels,
    # et les listes de colonnes numeriques/categorielles, via la fonction du module preprocessing.
    X, y, label_encoder, numeric_features, categorical_features = split_features_target(df)
    # Recupere la liste des noms de classes originaux (ex: ["eleve", "faible", "moyen"])
    # dans l'ordre correspondant aux valeurs encodees par label_encoder.
    class_names = list(label_encoder.classes_)

    # Divise les donnees en un ensemble d'entrainement (80%) et un ensemble de test (20%).
    # random_state garantit que ce decoupage est identique a chaque execution.
    # stratify=y garantit que la proportion de chaque classe est preservee
    # de facon similaire dans les ensembles d'entrainement et de test (important si les classes
    # sont desequilibrees).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Construit le ColumnTransformer de pretraitement (partage par tous les modeles),
    # a partir des listes de colonnes numeriques et categorielles identifiees precedemment.
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    # Construit le dictionnaire des modeles a comparer (regression logistique, random forest,
    # et XGBoost si disponible), chacun combine avec le meme pretraitement.
    models = build_models(preprocessor, n_classes=len(class_names))

    # Calcule un poids pour chaque observation de l'ensemble d'entrainement, de facon a
    # compenser le desequilibre des classes (les observations des classes minoritaires
    # recoivent un poids plus eleve). Utilise specifiquement pour XGBoost plus bas,
    # les deux autres modeles gerant deja le desequilibre via class_weight="balanced".
    train_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # Liste qui va accumuler les resultats (metriques) de chaque modele entraine.
    results = []
    # Dictionnaire qui va stocker les objets modeles deja entraines (pipelines complets),
    # afin de pouvoir recuperer le meilleur d'entre eux plus tard sans le reentrainer.
    fitted_models = {}

    # Boucle sur chaque paire (nom du modele, pipeline) du dictionnaire "models".
    for name, model in models.items():
        # Affiche dans la console le nom du modele en cours d'entrainement, pour suivre la progression.
        print(f"Entrainement : {name}")

        # Cas particulier : pour XGBoost, on transmet le poids par observation calcule plus haut,
        # afin de gerer le desequilibre des classes. La syntaxe "model__sample_weight" permet de
        # passer ce parametre specifiquement a l'etape nommee "model" a l'interieur du Pipeline.
        if name == "XGBoost":
            model.fit(X_train, y_train, model__sample_weight=train_sample_weight)
        # Pour les autres modeles (regression logistique, random forest), pas besoin de poids
        # explicite car ils gerent deja le desequilibre via leur parametre class_weight="balanced".
        else:
            model.fit(X_train, y_train)

        # Utilise le modele fraichement entraine pour predire les classes sur l'ensemble de test.
        y_pred = model.predict(X_test)
        # Calcule egalement les probabilites associees a chaque classe pour l'ensemble de test
        # (necessaire pour calculer le ROC AUC plus bas).
        y_proba = model.predict_proba(X_test)
        # Stocke le modele entraine dans le dictionnaire fitted_models, indexe par son nom.
        fitted_models[name] = model

        # Genere un rapport detaille (precision, rappel, f1-score) pour chaque classe.
        # labels=np.arange(len(class_names)) force l'ordre des classes a correspondre
        # exactement a celui de class_names, meme si certaines classes sont absentes de y_test.
        # zero_division=0 evite une erreur/avertissement si une classe n'a aucune prediction
        # (dans ce cas, la metrique correspondante est simplement mise a 0).
        # output_dict=True renvoie le rapport sous forme de dictionnaire Python (plus facile
        # a manipuler/sauvegarder) plutot que sous forme de texte brut.
        report = classification_report(
            y_test, y_pred, labels=np.arange(len(class_names)),
            target_names=class_names, zero_division=0, output_dict=True,
        )
        # Calcule la matrice de confusion (tableau croise predictions vs realite),
        # avec le meme ordre de classes que ci-dessus pour la coherence.
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

        # Ajoute a la liste "results" un dictionnaire regroupant toutes les metriques
        # calculees pour ce modele, afin de pouvoir les comparer plus tard.
        results.append(
            {
                "modele": name,
                # Proportion globale de predictions correctes.
                "accuracy": accuracy_score(y_test, y_pred),
                # Precision moyenne ponderee par le nombre d'observations de chaque classe
                # (tient compte du desequilibre des classes).
                "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                # Rappel (sensibilite) moyen pondere par le nombre d'observations de chaque classe.
                "rappel_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                # F1-score moyen pondere (compromis entre precision et rappel) : c'est cette
                # metrique qui servira a determiner le "meilleur" modele plus bas.
                "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                # Aire sous la courbe ROC en strategie "un contre tous" (ovr = one-vs-rest),
                # ponderee par la frequence de chaque classe. Mesure la capacite du modele
                # a bien separer les classes independamment du seuil de decision choisi.
                "roc_auc_ovr_weighted": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"),
                # Rapport de classification detaille (dictionnaire), conserve pour export ulterieur.
                "classification_report": report,
                # Matrice de confusion convertie en liste Python (plus facile a serialiser en JSON).
                "confusion_matrix": cm.tolist(),
            }
        )

    # Transforme la liste de dictionnaires "results" en DataFrame pandas,
    # utilise la colonne "modele" comme index (au lieu d'un simple numero de ligne),
    # puis trie les lignes par F1-score pondere decroissant (le meilleur modele en premier).
    results_df = pd.DataFrame(results).set_index("modele").sort_values("f1_weighted", ascending=False)
    # Recupere le nom du modele ayant obtenu le meilleur F1-score (premiere ligne apres le tri).
    best_model_name = results_df.index[0]
    # Recupere l'objet modele (pipeline entraine) correspondant, depuis le dictionnaire fitted_models.
    best_model = fitted_models[best_model_name]

    # Affiche un tableau recapitulatif comparant les differents modeles sur les metriques principales.
    print("\nComparaison des modeles (tries par F1 pondere) :")
    print(results_df[["accuracy", "precision_weighted", "rappel_weighted", "f1_weighted", "roc_auc_ovr_weighted"]])
    # Annonce clairement quel modele a ete retenu comme le meilleur.
    print(f"\nMeilleur modele retenu : {best_model_name}")

    # Sauvegarde le pipeline complet (pretraitement + modele) du meilleur modele sur disque,
    # au format .joblib, pour pouvoir le recharger plus tard sans reentrainer (voir predict.py).
    joblib.dump(best_model, outdir_path / "best_model.joblib")
    # Sauvegarde l'encodeur de labels, necessaire pour retraduire les predictions numeriques
    # en libelles textuels lisibles (ex: 0 -> "faible").
    joblib.dump(label_encoder, outdir_path / "label_encoder.joblib")
    # Sauvegarde la liste ordonnee des noms de colonnes utilisees a l'entrainement, afin de
    # garantir que les futures observations a predire respectent exactement le meme format.
    joblib.dump(list(X.columns), outdir_path / "feature_columns.joblib")
    # Sauvegarde egalement, dans un fichier separe, la distinction entre colonnes numeriques
    # et categorielles (utile pour du debogage ou une reutilisation future du pretraitement).
    joblib.dump(
        {"numeric_features": numeric_features, "categorical_features": categorical_features},
        outdir_path / "feature_types.joblib",
    )

    # Prepare un dictionnaire destine a etre exporte en JSON, contenant le nom du meilleur
    # modele et un espace reserve ("comparaison") pour les metriques de chaque modele.
    metrics_export = {"best_model": best_model_name, "comparaison": {}}
    # Remplit la cle "comparaison" avec, pour chaque modele (dans l'ordre trie par f1_weighted),
    # un sous-dictionnaire de ses metriques principales, converties en float Python standard
    # (necessaire car les valeurs numpy ne sont pas toujours serialisables directement en JSON).
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
    # Ajoute egalement la liste des noms de classes (utile pour interpreter les resultats
    # sans avoir a recharger l'encodeur de labels).
    metrics_export["class_names"] = class_names

    # Ouvre (ou cree) le fichier metrics.json en mode ecriture, avec encodage utf-8.
    with open(outdir_path / "metrics.json", "w", encoding="utf-8") as f:
        # Ecrit le dictionnaire metrics_export dans ce fichier au format JSON,
        # avec une indentation de 2 espaces (lisibilite) et sans forcer l'ASCII
        # (les caracteres accentues sont donc conserves tels quels).
        json.dump(metrics_export, f, ensure_ascii=False, indent=2)

    # Message final confirmant a l'utilisateur ou trouver le modele sauvegarde.
    print(f"\nModele sauvegarde dans {outdir_path}/best_model.joblib")


# Ce bloc ne s'execute que si le fichier est lance directement en ligne de commande,
# pas si le fichier est simplement importe comme module par un autre script.
if __name__ == "__main__":
    # Cree un objet qui va gerer les arguments passes en ligne de commande,
    # avec une description affichee si on lance "python train.py --help".
    parser = argparse.ArgumentParser(description="Entraine les modeles de prediction du risque de paludisme.")
    # Declare l'argument optionnel "--data" : chemin vers le CSV brut a utiliser pour l'entrainement.
    # Valeur par defaut : "data/raw/paludisme.csv" si non precise par l'utilisateur.
    parser.add_argument("--data", default="data/raw/paludisme.csv", help="Chemin du CSV brut")
    # Declare l'argument optionnel "--outdir" : dossier ou sauvegarder les fichiers produits
    # (modele, encodeur, metriques...). Valeur par defaut : "models".
    parser.add_argument("--outdir", default="models", help="Dossier de sortie pour les modeles")
    # Analyse les arguments effectivement passes par l'utilisateur dans le terminal.
    args = parser.parse_args()
    # Lance la fonction principale avec les chemins fournis (ou les valeurs par defaut).
    main(args.data, args.outdir)
