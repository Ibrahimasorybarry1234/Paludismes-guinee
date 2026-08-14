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

# Permet d'utiliser la syntaxe moderne des annotations de type (ex: "str | Path")
# meme sur des versions de Python anterieures a 3.10.
from __future__ import annotations

# Module standard pour gerer les arguments passes en ligne de commande (--models-dir, --json, ...)
import argparse

# Module standard pour lire/ecrire des donnees au format JSON.
import json

# Classe pratique pour manipuler des chemins de fichiers de facon portable (Windows/Linux/Mac).
from pathlib import Path

# Bibliotheque utilisee pour charger (deserialiser) les objets Python
# sauvegardes sur disque (ici : le modele entraine, l'encodeur, la liste des colonnes).
import joblib

# Bibliotheque de manipulation de donnees tabulaires (DataFrame).
import pandas as pd


# Definition d'une classe qui regroupe tout ce qu'il faut pour faire une prediction :
# le modele, l'encodeur de labels et la liste des colonnes attendues en entree.
class RiskPredictor:
    """Encapsule le pipeline sklearn entraine + l'encodeur de labels."""

    # Constructeur de la classe : s'execute automatiquement quand on fait RiskPredictor("models").
    # "models_dir" est le dossier contenant les fichiers .joblib du modele entraine.
    # Le type "str | Path" veut dire que ce parametre peut etre soit une chaine, soit un objet Path.
    # La valeur par defaut est "models" si rien n'est precise.
    def __init__(self, models_dir: str | Path = "models"):
        # Convertit models_dir (qui peut etre une simple chaine de caracteres) en objet Path,
        # ce qui permet ensuite d'utiliser l'operateur "/" pour construire des chemins de fichiers.
        models_dir = Path(models_dir)

        # Charge le pipeline scikit-learn deja entraine (pretraitement + modele de classification)
        # depuis le fichier "best_model.joblib" situe dans le dossier models_dir.
        self.model = joblib.load(models_dir / "best_model.joblib")

        # Charge l'encodeur de labels (ex: LabelEncoder de sklearn) qui a servi a transformer
        # les niveaux de risque (ex: "faible", "moyen", "eleve") en valeurs numeriques pendant l'entrainement.
        # On en a besoin ici pour faire l'operation inverse : chiffre -> texte.
        self.label_encoder = joblib.load(models_dir / "label_encoder.joblib")

        # Charge la liste (ou l'objet) contenant les noms exacts des colonnes/features
        # utilisees pour entrainer le modele. Cela garantit que les nouvelles observations
        # sont presentees au modele dans le meme format que les donnees d'entrainement.
        self.feature_columns = joblib.load(models_dir / "feature_columns.joblib")

    # Methode pour predire le risque a partir d'une seule observation (un dictionnaire Python).
    # Elle retourne un dictionnaire contenant la prediction et les probabilites par classe.
    def predict_one(self, observation: dict) -> dict:
        """Predit le niveau de risque pour une observation unique.

        `observation` doit contenir au minimum les colonnes utilisees a
        l'entrainement (voir models/feature_columns.joblib). Les colonnes
        manquantes sont laissees a NaN et gerees par l'imputer du pipeline.
        """
        # Construit un dictionnaire "row" qui contient une valeur pour chaque colonne attendue
        # par le modele (self.feature_columns). Si une colonne n'existe pas dans "observation",
        # observation.get(col) renverra None (valeur manquante), qui sera geree plus tard
        # par l'etape d'imputation du pipeline sklearn.
        row = {col: observation.get(col) for col in self.feature_columns}

        # Transforme le dictionnaire "row" en DataFrame pandas d'une seule ligne,
        # car les pipelines scikit-learn attendent generalement un tableau/DataFrame en entree,
        # pas un simple dictionnaire.
        X = pd.DataFrame([row])

        # Utilise le modele pour predire la classe (niveau de risque encode en chiffre)
        # de cette unique observation. predict() renvoie un tableau, on prend le premier
        # (et seul) element avec [0].
        pred_encoded = self.model.predict(X)[0]

        # Calcule les probabilites associees a chaque classe possible pour cette observation
        # (ex: [0.7, 0.2, 0.1] pour 3 niveaux de risque). predict_proba() renvoie un tableau
        # de probabilites par ligne, on prend la premiere (et seule) ligne avec [0].
        proba = self.model.predict_proba(X)[0]

        # Recupere la liste des noms de classes originaux (ex: ["eleve", "faible", "moyen"])
        # dans l'ordre correspondant aux colonnes de proba.
        classes = self.label_encoder.classes_

        # Construit et retourne le dictionnaire de resultat final.
        return {
            # Convertit la classe predite (un chiffre, ex: 1) en son libelle original
            # (ex: "moyen") grace a l'encodeur de labels. inverse_transform attend une liste,
            # d'ou [pred_encoded], et on recupere le premier element du resultat avec [0].
            "niveau_risque_predit": self.label_encoder.inverse_transform([pred_encoded])[0],

            # Construit un dictionnaire associant chaque nom de classe (cls) a sa probabilite (p),
            # convertie en float Python standard (float(p)) pour etre serialisable en JSON
            # (les types numpy ne sont pas toujours directement compatibles avec json.dumps).
            "probabilites": {cls: float(p) for cls, p in zip(classes, proba)},
        }

    # Methode pour predire le risque sur plusieurs observations a la fois,
    # fournies sous forme de DataFrame pandas (ex: un fichier CSV charge).
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predit le niveau de risque pour un DataFrame de plusieurs lignes."""
        # Identifie les colonnes attendues par le modele (self.feature_columns) qui
        # sont absentes du DataFrame fourni par l'utilisateur.
        missing = [c for c in self.feature_columns if c not in df.columns]

        # Pour chaque colonne manquante, on l'ajoute au DataFrame en la remplissant
        # de valeurs None (valeurs manquantes), qui seront gerees par l'imputer du pipeline.
        for col in missing:
            df[col] = None

        # Selectionne uniquement les colonnes utiles au modele, dans le bon ordre
        # (celui de self.feature_columns), et ignore les colonnes en trop.
        X = df[self.feature_columns]

        # Predit la classe (encodee en chiffre) pour chaque ligne du DataFrame X.
        pred_encoded = self.model.predict(X)

        # Calcule les probabilites de chaque classe pour chaque ligne du DataFrame X.
        proba = self.model.predict_proba(X)

        # Cree une copie du DataFrame original pour ne pas le modifier directement (bonne pratique),
        # et c'est sur cette copie qu'on va ajouter les colonnes de resultats.
        out = df.copy()

        # Ajoute une nouvelle colonne "niveau_risque_predit" contenant le libelle textuel
        # de la prediction pour chaque ligne (conversion chiffre -> texte via l'encodeur).
        out["niveau_risque_predit"] = self.label_encoder.inverse_transform(pred_encoded)

        # Pour chaque classe possible (ex: "faible", "moyen", "eleve"), ajoute une colonne
        # "proba_<nom_de_la_classe>" contenant la probabilite correspondante pour chaque ligne.
        # enumerate(...) donne a la fois l'indice i (position de la colonne dans "proba")
        # et le nom de la classe (cls).
        for i, cls in enumerate(self.label_encoder.classes_):
            out[f"proba_{cls}"] = proba[:, i]

        # Retourne le DataFrame enrichi des colonnes de prediction et de probabilites.
        return out


# Ce bloc ne s'execute que si le fichier est lance directement en ligne de commande
# (ex: "python predict.py ..."), pas si le fichier est simplement importe comme module.
if __name__ == "__main__":
    # Cree un objet qui va gerer les arguments passes en ligne de commande,
    # avec une description affichee si on lance "python predict.py --help".
    parser = argparse.ArgumentParser(description="Predit le niveau de risque de paludisme.")

    # Declare l'argument optionnel "--models-dir" : chemin vers le dossier du modele.
    # Si l'utilisateur ne le precise pas, la valeur par defaut sera "models".
    parser.add_argument("--models-dir", default="models")

    # Declare l'argument obligatoire "--json" : une chaine de caracteres au format JSON
    # representant l'observation a predire. required=True signifie que le programme
    # s'arretera avec une erreur si cet argument n'est pas fourni.
    parser.add_argument("--json", required=True, help="Observation au format JSON")

    # Analyse les arguments effectivement passes par l'utilisateur dans le terminal
    # et les stocke dans l'objet "args" (ex: args.models_dir, args.json).
    args = parser.parse_args()

    # Instancie le predicteur en chargeant le modele depuis le dossier indique.
    predictor = RiskPredictor(args.models_dir)

    # Convertit la chaine JSON fournie en ligne de commande en un dictionnaire Python.
    observation = json.loads(args.json)

    # Appelle la methode de prediction sur cette observation unique.
    result = predictor.predict_one(observation)

    # Affiche le resultat dans le terminal au format JSON, joliment indente (indent=2),
    # en conservant les caracteres accentues/speciaux tels quels (ensure_ascii=False).
    print(json.dumps(result, ensure_ascii=False, indent=2))
