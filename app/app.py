"""
Application Streamlit — Prédiction du risque de paludisme en Guinée.

Lancement local :
    streamlit run app/app.py
"""
# On active les annotations de type "différées" : permet d'écrire des types
# comme "RiskPredictor | None" même sur des versions de Python plus anciennes
# que 3.10, car les annotations ne sont évaluées qu'à la demande.
from __future__ import annotations

# Module standard pour lire/écrire du JSON (utilisé pour metrics.json).
import json
# Module standard pour manipuler le chemin d'exécution Python (sys.path),
# afin de pouvoir importer des modules situés dans le dossier "src/".
import sys
# Classe orientée objet pour manipuler des chemins de fichiers de façon
# portable (Windows/Linux/Mac), plutôt que de concaténer des chaînes.
from pathlib import Path

# Bibliothèque pour charger un modèle scikit-learn/XGBoost sérialisé (.joblib).
# Elle n'est pas utilisée directement ici (RiskPredictor s'en charge en interne),
# mais elle est importée au cas où elle serait nécessaire dans ce fichier.
import joblib
# Bibliothèque de manipulation de données tabulaires (DataFrame).
import pandas as pd
# Sous-module de Plotly Express pour créer rapidement des graphiques interactifs.
import plotly.express as px
# Framework principal utilisé pour construire l'interface web de l'application.
import streamlit as st

# Import de fonctions/constantes définies dans un fichier "theme.py" du même
# dossier : habillage visuel (couleurs, bannière, mise en forme des résultats).
from theme import apply_theme, hero_banner, result_card, style_fig, RISK_COLORS

# Calcule le dossier racine du projet : on part du fichier courant (__file__),
# on résout les liens symboliques éventuels (resolve()), on remonte au dossier
# parent (le dossier "app/"), puis encore au parent (la racine du projet).
ROOT_DIR = Path(__file__).resolve().parent.parent
# Ajoute le dossier "src/" du projet au chemin de recherche des modules Python,
# afin de pouvoir faire "from predict import RiskPredictor" juste après,
# même si "src/" n'est pas un package installé.
sys.path.insert(0, str(ROOT_DIR / "src"))
# Import de la classe qui encapsule le chargement du modèle et la prédiction.
# "# noqa: E402" désactive l'avertissement de style "import pas en haut du
# fichier", ici justifié car on doit d'abord modifier sys.path.
from predict import RiskPredictor  # noqa: E402

# Chemin vers le dossier contenant les modèles entraînés (best_model.joblib, etc.).
MODELS_DIR = ROOT_DIR / "models"
# Chemin vers le fichier CSV contenant les données brutes utilisées pour
# l'exploration des données.
DATA_PATH = ROOT_DIR / "data" / "raw" / "paludisme.csv"

# Dictionnaire associant chaque région administrative de Guinée à la liste
# de ses préfectures. Sert à générer dynamiquement les menus déroulants
# "Région" -> "Préfecture" dans le formulaire de prédiction.
REGIONS_PREFECTURES = {
    "Boké": ["Boffa", "Boké", "Fria", "Gaoual", "Koundara"],
    "Conakry": ["Conakry"],
    "Faranah": ["Dabola", "Dinguiraye", "Faranah", "Kissidougou"],
    "Kankan": ["Kankan", "Kouroussa", "Kérouané", "Mandiana", "Siguiri"],
    "Kindia": ["Coyah", "Dubréka", "Forécariah", "Kindia", "Télimélé"],
    "Labé": ["Koubia", "Labé", "Lélouma", "Mali", "Tougué"],
    "Mamou": ["Dalaba", "Mamou", "Pita"],
    "Nzérékoré": ["Beyla", "Guéckédou", "Lola", "Macenta", "Nzérékoré", "Yomou"],
}
# Liste des zones climatiques possibles, utilisée pour peupler un menu déroulant.
ZONES_CLIMATIQUES = ["côtier", "forestier", "montagneux", "soudano-guinéen"]

# RISK_COLORS vient maintenant de theme.py (vert forêt / or / rouge risque),
# pour rester cohérent avec .streamlit/config.toml et le reste de l'identité.

# Configuration globale de la page Streamlit : doit être appelé une seule
# fois, avant tout autre appel st.*, sinon Streamlit lève une erreur.
st.set_page_config(
    page_title="Paludisme Guinée — Prédiction du risque",  # titre affiché dans l'onglet du navigateur
    page_icon="🦟",  # icône (emoji) affichée dans l'onglet du navigateur
    layout="wide",  # utilise toute la largeur de l'écran plutôt qu'une colonne centrée
)
# Applique le thème visuel personnalisé défini dans theme.py (couleurs, styles CSS...).
apply_theme()


# Décorateur Streamlit qui met en cache la RESSOURCE retournée par la fonction
# (ici un objet RiskPredictor, potentiellement lourd/non sérialisable) :
# la fonction ne sera exécutée qu'une seule fois tant que ses paramètres et
# son code ne changent pas, ce qui évite de recharger le modèle à chaque interaction.
@st.cache_resource
def load_predictor() -> RiskPredictor | None:
    # Si le fichier du modèle entraîné n'existe pas encore sur le disque...
    if not (MODELS_DIR / "best_model.joblib").exists():
        # ...on retourne None : l'application saura qu'aucun modèle n'est disponible.
        return None
    # Sinon, on instancie et retourne le prédicteur, initialisé avec le
    # dossier contenant les fichiers du modèle.
    return RiskPredictor(MODELS_DIR)


# Décorateur Streamlit qui met en cache les DONNÉES retournées (ici un
# DataFrame) : évite de relire le CSV depuis le disque à chaque interaction
# utilisateur, tant que le fichier ou le code de la fonction ne change pas.
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    # Lit le CSV en utilisant l'encodage "utf-8-sig", qui gère correctement
    # un éventuel BOM (marqueur d'ordre des octets) ajouté par certains
    # outils (par ex. Excel) lors de l'export du fichier.
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


# Même logique de cache que ci-dessus, appliquée cette fois au chargement
# des métriques de performance du modèle.
@st.cache_data
def load_metrics() -> dict:
    # Construit le chemin vers le fichier JSON des métriques.
    metrics_path = MODELS_DIR / "metrics.json"
    # Si ce fichier n'existe pas (modèle pas encore entraîné)...
    if not metrics_path.exists():
        # ...on retourne un dictionnaire vide plutôt que de planter.
        return {}
    # Ouvre le fichier en lecture, avec un encodage explicite pour bien
    # gérer les caractères accentués français.
    with open(metrics_path, encoding="utf-8") as f:
        # Parse le contenu JSON et le retourne sous forme de dictionnaire Python.
        return json.load(f)


# Fonction qui construit et affiche l'onglet "Prédiction" de l'application.
# Elle prend en paramètre le prédicteur (ou None s'il n'existe pas encore).
def page_prediction(predictor: RiskPredictor | None) -> None:
    # Titre de section affiché en haut de la page.
    st.header("Prédire le niveau de risque")
    # Texte descriptif secondaire, affiché en plus petit sous le titre.
    st.caption(
        "Renseignez les conditions météorologiques et environnementales d'une "
        "zone pour estimer le niveau de risque de paludisme (faible / moyen / élevé)."
    )

    # Si aucun modèle n'a été chargé (fichier manquant)...
    if predictor is None:
        # ...on affiche un message d'erreur expliquant comment générer le modèle,
        # avec des extraits de code mis en forme via des backticks Markdown.
        st.error(
            "Aucun modèle entraîné trouvé dans `models/`. "
            "Exécutez d'abord `python src/train.py` pour générer `models/best_model.joblib`."
        )
        # On arrête l'exécution de la fonction ici : impossible de continuer
        # sans modèle chargé.
        return

    # Pas de st.form ici : à l'intérieur d'un formulaire, les widgets ne se
    # mettent à jour qu'à la soumission, donc la préfecture ne suivrait pas
    # le changement de région tant qu'on n'a pas cliqué sur le bouton. On
    # utilise à la place un bouton classique, avec les 3 blocs alignés sur
    # la même ligne.
    # Crée 3 colonnes de largeur égale pour organiser le formulaire horizontalement.
    col1, col2, col3 = st.columns(3)

    # Bloc de widgets affiché dans la première colonne.
    with col1:
        # Sous-titre de la colonne.
        st.subheader("Localisation")
        # Menu déroulant pour choisir la région, parmi les clés du dictionnaire
        # REGIONS_PREFECTURES. "key" donne un identifiant unique au widget
        # dans la session Streamlit (utile pour retenir sa valeur entre les réexécutions).
        region = st.selectbox("Région administrative", list(REGIONS_PREFECTURES.keys()), key="region_select")
        # Menu déroulant pour la préfecture, dont les options dépendent de la
        # région sélectionnée. La "key" inclut le nom de la région : cela force
        # Streamlit à recréer/réinitialiser ce widget quand la région change,
        # ce qui garantit que la liste de préfectures affichée reste cohérente.
        prefecture = st.selectbox("Préfecture", REGIONS_PREFECTURES[region], key=f"prefecture_select_{region}")
        # Menu déroulant pour la zone climatique.
        zone_climatique = st.selectbox("Zone climatique", ZONES_CLIMATIQUES, key="zone_select")
        # Curseur (slider) pour choisir le mois, entre 1 et 12, valeur par défaut 8.
        mois = st.slider("Mois", 1, 12, 8, key="mois_select")
        # Curseur pour choisir la semaine de l'année, entre 1 et 53, valeur par défaut 32.
        semaine = st.slider("Semaine de l'année", 1, 53, 32, key="semaine_select")

    # Bloc de widgets affiché dans la deuxième colonne : variables météo.
    with col2:
        st.subheader("Météo")
        # Chaque slider ci-dessous suit le schéma :
        # st.slider(label, valeur_min, valeur_max, valeur_par_défaut, key=...)
        temperature_moyenne_c = st.slider("Température moyenne (°C)", 20.0, 34.0, 27.0, key="temp_moy")
        temperature_min_c = st.slider("Température min (°C)", 15.0, 30.0, 22.5, key="temp_min")
        temperature_max_c = st.slider("Température max (°C)", 24.0, 39.0, 31.5, key="temp_max")
        precipitation_mm = st.slider("Précipitation du jour (mm)", 0.0, 250.0, 13.0, key="precip_jour")
        precipitation_7j_mm = st.slider("Précipitation cumulée 7j (mm)", 0.0, 600.0, 90.0, key="precip_7j")
        precipitation_30j_mm = st.slider("Précipitation cumulée 30j (mm)", 0.0, 1600.0, 375.0, key="precip_30j")

    # Bloc de widgets affiché dans la troisième colonne : variables environnementales.
    with col3:
        st.subheader("Environnement")
        humidite_relative_pct = st.slider("Humidité relative (%)", 40.0, 100.0, 70.0, key="humid_rel")
        vitesse_vent_kmh = st.slider("Vitesse du vent (km/h)", 0.0, 20.0, 9.0, key="vent")
        # Ce slider est en entiers (bornes 0 et 7, pas de décimales) car le
        # nombre de jours de pluie est forcément un entier.
        jours_pluie_7j = st.slider("Jours de pluie (7 derniers jours)", 0, 7, 4, key="jours_pluie")
        humidite_sol_pct = st.slider("Humidité du sol (%)", 20.0, 100.0, 62.0, key="humid_sol")
        # Indice normalisé entre 0 et 1 représentant la présence d'eau stagnante.
        indice_eau_stagnante = st.slider("Indice d'eau stagnante (0-1)", 0.0, 1.0, 0.6, key="eau_stagnante")

    # Bouton principal de soumission. "type='primary'" lui donne le style mis
    # en avant (couleur d'accent), "use_container_width=True" l'étire sur
    # toute la largeur disponible. La variable "submitted" vaut True
    # uniquement lors du run qui suit immédiatement le clic.
    submitted = st.button("Prédire le niveau de risque", type="primary", use_container_width=True)

    # Si l'utilisateur vient de cliquer sur le bouton...
    if submitted:
        # Coordonnées approximatives = moyenne nationale (non demandées à l'utilisateur)
        # On construit un dictionnaire regroupant toutes les variables saisies,
        # avec les mêmes noms de colonnes que ceux attendus par le modèle entraîné.
        observation = {
            "mois": mois,
            "semaine": semaine,
            "region_administrative": region,
            "prefecture": prefecture,
            "zone_climatique": zone_climatique,
            # Latitude/longitude fixes car non demandées à l'utilisateur :
            # on utilise une position approximative représentant la moyenne nationale.
            "latitude": 10.22,
            "longitude": -11.44,
            "temperature_moyenne_c": temperature_moyenne_c,
            "temperature_min_c": temperature_min_c,
            "temperature_max_c": temperature_max_c,
            "precipitation_mm": precipitation_mm,
            "precipitation_7j_mm": precipitation_7j_mm,
            "precipitation_30j_mm": precipitation_30j_mm,
            "humidite_relative_pct": humidite_relative_pct,
            "vitesse_vent_kmh": vitesse_vent_kmh,
            "jours_pluie_7j": jours_pluie_7j,
            "humidite_sol_pct": humidite_sol_pct,
            "indice_eau_stagnante": indice_eau_stagnante,
            # Approximation grossière du "jour de l'année" à partir du mois
            # (30 jours par mois en moyenne), utilisée comme variable numérique
            # supplémentaire pour le modèle.
            "jour_annee": mois * 30,
        }

        # Appelle la méthode de prédiction du modèle avec cette observation
        # unique ; retourne un dictionnaire contenant le résultat.
        result = predictor.predict_one(observation)
        # Extrait le niveau de risque prédit (ex. "faible", "moyen", "eleve").
        niveau = result["niveau_risque_predit"]
        # Extrait le dictionnaire des probabilités associées à chaque classe.
        proba = result["probabilites"]

        # Affiche une ligne de séparation horizontale avant les résultats.
        st.divider()
        # Affiche une "carte" HTML personnalisée (définie dans theme.py) qui
        # met en valeur le niveau de risque prédit. "unsafe_allow_html=True"
        # est nécessaire car result_card() retourne du HTML brut.
        st.markdown(result_card(niveau), unsafe_allow_html=True)

        # Construit un DataFrame à partir du dictionnaire de probabilités,
        # avec une colonne "niveau_risque" (les clés) et une colonne
        # "probabilite" (les valeurs), pour pouvoir le tracer avec Plotly.
        proba_df = pd.DataFrame({"niveau_risque": list(proba.keys()), "probabilite": list(proba.values())})
        # Ordre logique des niveaux de risque, du plus faible au plus élevé.
        order = ["faible", "moyen", "eleve"]
        # Convertit la colonne en type "catégoriel ordonné" selon cet ordre,
        # afin que le graphique affiche les barres dans le bon sens plutôt
        # que dans un ordre alphabétique ou aléatoire.
        proba_df["niveau_risque"] = pd.Categorical(proba_df["niveau_risque"], categories=order, ordered=True)
        # Trie le DataFrame selon cet ordre catégoriel.
        proba_df = proba_df.sort_values("niveau_risque")

        # Crée un graphique en barres des probabilités par niveau de risque.
        fig = px.bar(
            proba_df, x="niveau_risque", y="probabilite", color="niveau_risque",
            # Utilise le mapping de couleurs cohérent défini dans theme.py.
            color_discrete_map=RISK_COLORS,
            # Affiche automatiquement la valeur de chaque barre, formatée en
            # pourcentage avec une décimale (ex. "42.3%").
            text_auto=".1%",
            # Renomme les axes pour l'affichage (labels plus lisibles).
            labels={"niveau_risque": "Niveau de risque", "probabilite": "Probabilité"},
        )
        # Masque la légende (redondante puisque les couleurs sont déjà sur
        # l'axe X) et formate l'axe Y en pourcentage sans décimale.
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        # Affiche le graphique dans Streamlit, en appliquant d'abord le style
        # visuel personnalisé (style_fig), et en l'étirant sur toute la largeur.
        st.plotly_chart(style_fig(fig), use_container_width=True)

        # Si le niveau de risque prédit est "élevé"...
        if niveau == "eleve":
            # ...on affiche un avertissement supplémentaire avec des
            # recommandations d'action pour les autorités sanitaires.
            st.warning(
                "⚠️ Risque élevé : conditions favorables à la prolifération des vecteurs "
                "(eau stagnante, humidité, précipitations). Renforcer la sensibilisation et "
                "la disponibilité des traitements dans les centres de santé concernés."
            )


# Fonction qui construit et affiche l'onglet "Exploration des données".
# Prend en paramètre le DataFrame complet des données brutes.
def page_exploration(df: pd.DataFrame) -> None:
    st.header("Exploration des données")
    # Affiche un résumé du nombre total d'observations et du nombre de
    # régions distinctes présentes dans les données. "{:,}" formate le
    # nombre avec des séparateurs de milliers.
    st.caption(f"{len(df):,} observations couvrant les {df['region_administrative'].nunique()} régions administratives de Guinée.")

    # Crée deux colonnes de largeur égale (ratio 1:1) pour afficher deux
    # graphiques côte à côte.
    col1, col2 = st.columns([1, 1])
    with col1:
        # Compte le nombre d'observations par niveau de risque, puis réordonne
        # les résultats selon l'ordre logique faible/moyen/élevé (au lieu de
        # l'ordre par défaut, qui serait par fréquence décroissante).
        risk_counts = df["niveau_risque"].value_counts().reindex(["faible", "moyen", "eleve"]).reset_index()
        # Renomme les colonnes du DataFrame résultant pour plus de clarté.
        risk_counts.columns = ["niveau_risque", "nombre"]
        # Graphique en barres du nombre d'observations par niveau de risque.
        fig = px.bar(
            risk_counts, x="niveau_risque", y="nombre", color="niveau_risque",
            color_discrete_map=RISK_COLORS, title="Répartition du niveau de risque",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with col2:
        # Tableau croisé (crosstab) entre région et niveau de risque, avec
        # "normalize='index'" pour obtenir des proportions (chaque ligne/région
        # somme à 1) plutôt que des comptages bruts.
        region_risk = pd.crosstab(df["region_administrative"], df["niveau_risque"], normalize="index")
        # S'assure que les 3 colonnes de niveaux de risque existent toutes et
        # sont dans le bon ordre, en remplissant par 0 celles qui manqueraient.
        region_risk = region_risk.reindex(columns=["faible", "moyen", "eleve"], fill_value=0).reset_index()
        # Transforme le tableau du format "large" (une colonne par niveau de
        # risque) au format "long" (une ligne par combinaison région/niveau),
        # format requis par Plotly Express pour un barplot empilé multi-catégories.
        region_risk_melted = region_risk.melt(id_vars="region_administrative", var_name="niveau_risque", value_name="part")
        # Graphique en barres empilées montrant la part de chaque niveau de
        # risque au sein de chaque région.
        fig2 = px.bar(
            region_risk_melted, x="region_administrative", y="part", color="niveau_risque",
            color_discrete_map=RISK_COLORS, title="Part du risque par région",
            barmode="stack",
        )
        # Affiche l'axe Y en pourcentage sans décimale.
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    st.subheader("Relation météo / risque")
    # Menu déroulant permettant à l'utilisateur de choisir quelle variable
    # météo/environnementale explorer par rapport au niveau de risque.
    weather_col = st.selectbox(
        "Variable météo",
        ["temperature_moyenne_c", "precipitation_7j_mm", "precipitation_30j_mm",
         "humidite_relative_pct", "humidite_sol_pct", "indice_eau_stagnante"],
    )
    # Histogramme de la variable choisie, avec une courbe/distribution par
    # niveau de risque superposée ("overlay") et rendue semi-transparente
    # (opacity=0.6) pour permettre de comparer les distributions.
    # "histnorm='probability density'" normalise chaque histogramme pour
    # que les groupes de tailles différentes restent comparables.
    fig3 = px.histogram(
        df, x=weather_col, color="niveau_risque", barmode="overlay",
        color_discrete_map=RISK_COLORS, histnorm="probability density", opacity=0.6,
    )
    st.plotly_chart(style_fig(fig3), use_container_width=True)

    # Section repliable (accordéon) contenant un échantillon brut des données.
    with st.expander("Voir un échantillon des données brutes"):
        # Affiche un échantillon aléatoire de 200 lignes maximum (ou moins si
        # le jeu de données en contient moins). "random_state=42" fixe la
        # graine aléatoire pour que l'échantillon soit reproductible d'un
        # rechargement à l'autre.
        st.dataframe(df.sample(min(200, len(df)), random_state=42), use_container_width=True)


# Fonction qui construit et affiche l'onglet "Performance du modèle".
# Prend en paramètre le dictionnaire de métriques chargé depuis metrics.json.
def page_modele(metrics: dict) -> None:
    st.header("Performance du modèle")
    # Si le dictionnaire de métriques est vide (fichier absent)...
    if not metrics:
        # ...on informe l'utilisateur qu'il faut d'abord entraîner un modèle.
        st.info("Aucune métrique disponible. Exécutez `python src/train.py` pour entraîner les modèles.")
        # On arrête ici : rien d'autre à afficher sans métriques.
        return

    # Affiche le nom du meilleur modèle retenu, en gras via Markdown ("**...**").
    st.caption(f"Modèle retenu : **{metrics['best_model']}** (meilleur F1 pondéré sur le jeu de test).")

    # Convertit le sous-dictionnaire "comparaison" (un modèle -> ses métriques)
    # en DataFrame, puis le transpose (.T) pour avoir un modèle par ligne et
    # une métrique par colonne (plus lisible pour un tableau).
    comp_df = pd.DataFrame(metrics["comparaison"]).T
    # Trie les modèles du meilleur au moins bon selon le F1 pondéré.
    comp_df = comp_df.sort_values("f1_weighted", ascending=False)
    # Affiche le tableau, avec chaque valeur numérique formatée à 3 décimales.
    st.dataframe(comp_df.style.format("{:.3f}"), use_container_width=True)

    # Graphique en barres groupées comparant plusieurs métriques (accuracy,
    # précision, rappel, F1, ROC AUC) pour chaque modèle testé.
    fig = px.bar(
        # reset_index() remet le nom du modèle comme colonne normale
        # (plutôt que comme index), puis on la renomme "modele" pour la clarté.
        comp_df.reset_index().rename(columns={"index": "modele"}),
        x="modele", y=["accuracy", "precision_weighted", "rappel_weighted", "f1_weighted", "roc_auc_ovr_weighted"],
        barmode="group", title="Comparaison des modèles",
    )
    # Fixe l'échelle de l'axe Y entre 0 et 1 (toutes ces métriques sont des
    # proportions), et renomme le titre de la légende.
    fig.update_layout(yaxis_range=[0, 1], legend_title="Métrique")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # Bloc de texte Markdown expliquant la méthodologie utilisée pour
    # entraîner et évaluer les modèles, affiché sous forme de liste à puces.
    st.markdown(
        """
        **Méthodologie (résumé) :**
        - Cible : `niveau_risque` (faible / moyen / élevé), encodée avec `LabelEncoder`.
        - Variables exclues pour éviter la fuite de données : identifiant, date brute,
          année, et les indicateurs sanitaires simulés directement corrélés à la cible.
        - Prétraitement : imputation (médiane / mode) + standardisation des variables
          numériques + encodage One-Hot des variables catégorielles.
        - Séparation stratifiée train/test (80/20), pondération des classes pour gérer
          le déséquilibre.
        - Trois modèles comparés : Régression logistique, Random Forest, XGBoost.
        """
    )


# Fonction principale : point d'entrée de l'application, orchestre l'affichage
# de la bannière, le chargement des données/modèle, et les trois onglets.
def main() -> None:
    # Affiche la bannière d'en-tête personnalisée (définie dans theme.py),
    # avec un petit texte d'introduction, un titre principal et un sous-titre.
    hero_banner(
        eyebrow="Ministère de la Santé — Surveillance épidémiologique",
        title="Prédiction du risque de paludisme en Guinée",
        subtitle=(
            "Projet data science — données synthétiques à but pédagogique. "
            "Ne remplace pas les données de surveillance épidémiologique officielles."
        ),
    )

    # Charge (ou récupère depuis le cache) le prédicteur, les données brutes
    # et les métriques du modèle. Ces trois appels ne rechargent réellement
    # les ressources que si elles n'ont pas déjà été mises en cache.
    predictor = load_predictor()
    df = load_raw_data()
    metrics = load_metrics()

    # Crée trois onglets dans l'interface, avec leurs libellés respectifs.
    tab1, tab2, tab3 = st.tabs(["Prédiction", "Exploration des données", "Performance du modèle"])
    # Tout ce qui est affiché à l'intérieur de ce bloc "with" apparaît dans
    # le premier onglet ("Prédiction").
    with tab1:
        page_prediction(predictor)
    # Contenu du deuxième onglet ("Exploration des données").
    with tab2:
        page_exploration(df)
    # Contenu du troisième onglet ("Performance du modèle").
    with tab3:
        page_modele(metrics)

    # Ajoute un titre de section dans la barre latérale (sidebar).
    st.sidebar.header("À propos")
    # Ajoute un texte explicatif dans la barre latérale, décrivant l'objectif
    # général de l'application.
    st.sidebar.write(
        "Cette application estime le niveau de risque de paludisme "
        "(faible / moyen / élevé) à partir de variables météorologiques "
        "et environnementales, pour aider à anticiper les périodes de "
        "forte demande dans les centres de santé."
    )
    # Ajoute un avertissement dans la barre latérale, rappelant que les
    # données utilisées sont synthétiques et non opérationnelles.
    st.sidebar.warning(
        "⚠️ Les données utilisées sont **synthétiques**. Une utilisation "
        "opérationnelle nécessiterait des données validées (OMS/HDX, "
        "services de santé, stations météo)."
    )


# Point d'entrée standard en Python : ce bloc ne s'exécute que si le fichier
# est lancé directement (par ex. via "streamlit run app/app.py"), et non
# lorsqu'il est importé comme module par un autre script.
if __name__ == "__main__":
    main()
