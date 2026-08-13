# 🦟 Paludisme Guinée — Prédiction du niveau de risque

Projet de data science (démarche CRISP-DM) visant à estimer le **niveau de
risque de paludisme** (`faible` / `moyen` / `eleve`) dans les régions de
Guinée à partir de variables météorologiques et environnementales, afin
d'aider les centres de santé à anticiper les périodes de forte demande.

> ⚠️ **Données synthétiques.** Le fichier `data/raw/paludisme.csv` est un
> jeu de données simulé, généré à des fins pédagogiques. Une utilisation
> opérationnelle réelle nécessiterait des données validées par les services
> de santé (OMS, HDX, stations météo officielles).

## Sommaire

- [Aperçu](#aperçu)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Reproduire le projet](#reproduire-le-projet)
- [Lancer l'application Streamlit](#lancer-lapplication-streamlit)
- [Résultats](#résultats)
- [Déploiement](#déploiement)
- [Licence](#licence)

## Aperçu

- **Cible :** `niveau_risque` (3 classes : faible, moyen, eleve)
- **Variables explicatives :** température, précipitations (jour / 7j / 30j),
  humidité relative et du sol, vent, jours de pluie, indice d'eau
  stagnante, région/préfecture, zone climatique, mois/semaine.
- **Fuite de données évitée :** les colonnes `cas_paludisme_simules` et
  `taux_incidence_simule_pour_1000` (directement dérivées de la cible),
  ainsi que `id_observation`, `date` brute et `annee`, sont exclues des
  variables explicatives.
- **Modèles comparés :** Régression logistique, Random Forest, XGBoost.
- **Métrique de sélection :** F1 pondéré (le rappel de la classe `eleve`
  est particulièrement surveillé, l'objectif étant de ne pas manquer une
  période à risque).

## Structure du dépôt

```
Paludismes-guinee/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/                   # données brutes (paludisme.csv)
│   └── processed/             # données nettoyées (générées par src/train.py)
├── notebooks/
│   ├── 01_analyse_exploratoire.ipynb
│   └── 02_modelisation.ipynb
├── src/
│   ├── preprocessing.py       # chargement + pipeline de prétraitement
│   ├── train.py                # entrainement + comparaison des 3 modèles
│   └── predict.py              # inference (module RiskPredictor)
├── models/                    # modèles sérialisés (.joblib) + metrics.json
├── app/
│   └── app.py                  # application Streamlit
├── rapport/
│   └── rapport.pdf             # rapport complet (13 pages)
├── presentation/
│   └── presentation.pptx       # support de présentation du projet
└── .github/workflows/ci.yml    # pipeline CI/CD GitHub Actions
```

## Installation

Prérequis : Python 3.10+.

```bash
git clone https://github.com/Ibrahimasorybarry1234/Paludismes-guinee.git
cd Paludismes-guinee
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduire le projet

1. **Analyse exploratoire** : ouvrir `notebooks/01_analyse_exploratoire.ipynb`
   dans Jupyter (le notebook charge automatiquement `data/raw/paludisme.csv`).
2. **Entrainement des modèles** (régénère `models/*.joblib`,
   `models/metrics.json` et `data/processed/paludisme_clean.csv`) :

   ```bash
   python src/train.py --data data/raw/paludisme.csv --outdir models
   ```

3. **Modélisation détaillée** (mêmes étapes, format notebook, avec
   matrices de confusion et importance des variables) :
   `notebooks/02_modelisation.ipynb`.
4. **Prédiction en ligne de commande** :

   ```bash
   python src/predict.py --json '{"mois": 8, "region_administrative": "Nzérékoré", "prefecture": "Nzérékoré", "zone_climatique": "forestier", "temperature_moyenne_c": 26.5, "precipitation_7j_mm": 90, "humidite_sol_pct": 70, "indice_eau_stagnante": 0.8}'
   ```

## Lancer l'application Streamlit

```bash
streamlit run app/app.py
```

L'application propose trois onglets :
- **Prédiction** : formulaire interactif pour estimer le niveau de risque
  d'une nouvelle observation, avec probabilités par classe.
- **Exploration des données** : répartition du risque, comparaison par
  région, distributions météo par niveau de risque.
- **Performance du modèle** : comparaison des 3 modèles (accuracy,
  précision, rappel, F1, ROC AUC).

## Résultats

Les métriques exactes sont régénérées dans `models/metrics.json` à chaque
exécution de `src/train.py` (résultats indicatifs sur le jeu de test,
20 % des données, séparation stratifiée) :

| Modèle | Accuracy | F1 pondéré | ROC AUC (OvR) |
|---|---|---|---|
| Régression logistique | ~0.93 | ~0.93 | ~0.99 |
| Random Forest | ~0.92 | ~0.92 | ~0.99 |
| XGBoost | voir `models/metrics.json` après entrainement local |

> Le modèle XGBoost nécessite le paquet `xgboost` (inclus dans
> `requirements.txt`). S'il n'est pas installé, `src/train.py` l'ignore
> automatiquement et compare uniquement les deux autres modèles.

## Déploiement

### Local (obligatoire)

```bash
streamlit run app/app.py
```

### En ligne (bonus)

- **Streamlit Community Cloud** : connecter ce dépôt GitHub sur
  [share.streamlit.io](https://share.streamlit.io), pointer sur `app/app.py`,
  branche `main`.
- **Hugging Face Spaces** : créer un Space de type *Streamlit*, y pousser
  ce dépôt (ou le lier via GitHub Actions).
- **AWS EC2** : installer les dépendances (`requirements.txt`) et lancer
  `streamlit run app/app.py --server.port 80 --server.address 0.0.0.0`
  derrière un groupe de sécurité autorisant le port choisi.

Le pipeline `.github/workflows/ci.yml` vérifie à chaque push/PR sur `main`
que le code s'importe correctement et que l'entrainement s'exécute sans
erreur, puis publie le modèle entraîné comme artefact téléchargeable.

## Licence

Ce projet est distribué sous licence [MIT](LICENSE).

## Auteur

Ibrahima Sory Barry
