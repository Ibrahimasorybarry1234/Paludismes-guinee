"""
Application Streamlit — Prédiction du risque de paludisme en Guinée.

Lancement local :
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from predict import RiskPredictor  # noqa: E402

MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "raw" / "paludisme.csv"

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
ZONES_CLIMATIQUES = ["côtier", "forestier", "montagneux", "soudano-guinéen"]

RISK_COLORS = {"faible": "#2e7d32", "moyen": "#f9a825", "eleve": "#c62828"}

st.set_page_config(
    page_title="Paludisme Guinée — Prédiction du risque",
    page_icon="🦟",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> RiskPredictor | None:
    if not (MODELS_DIR / "best_model.joblib").exists():
        return None
    return RiskPredictor(MODELS_DIR)


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


@st.cache_data
def load_metrics() -> dict:
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def page_prediction(predictor: RiskPredictor | None) -> None:
    st.header("🔮 Prédire le niveau de risque")
    st.caption(
        "Renseignez les conditions météorologiques et environnementales d'une "
        "zone pour estimer le niveau de risque de paludisme (faible / moyen / élevé)."
    )

    if predictor is None:
        st.error(
            "Aucun modèle entraîné trouvé dans `models/`. "
            "Exécutez d'abord `python src/train.py` pour générer `models/best_model.joblib`."
        )
        return

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Localisation")
            region = st.selectbox("Région administrative", list(REGIONS_PREFECTURES.keys()))
            prefecture = st.selectbox("Préfecture", REGIONS_PREFECTURES[region])
            zone_climatique = st.selectbox("Zone climatique", ZONES_CLIMATIQUES)
            mois = st.slider("Mois", 1, 12, 8)
            semaine = st.slider("Semaine de l'année", 1, 53, 32)

        with col2:
            st.subheader("Météo")
            temperature_moyenne_c = st.slider("Température moyenne (°C)", 20.0, 34.0, 27.0)
            temperature_min_c = st.slider("Température min (°C)", 15.0, 30.0, 22.5)
            temperature_max_c = st.slider("Température max (°C)", 24.0, 39.0, 31.5)
            precipitation_mm = st.slider("Précipitation du jour (mm)", 0.0, 250.0, 13.0)
            precipitation_7j_mm = st.slider("Précipitation cumulée 7j (mm)", 0.0, 600.0, 90.0)
            precipitation_30j_mm = st.slider("Précipitation cumulée 30j (mm)", 0.0, 1600.0, 375.0)

        with col3:
            st.subheader("Environnement")
            humidite_relative_pct = st.slider("Humidité relative (%)", 40.0, 100.0, 70.0)
            vitesse_vent_kmh = st.slider("Vitesse du vent (km/h)", 0.0, 20.0, 9.0)
            jours_pluie_7j = st.slider("Jours de pluie (7 derniers jours)", 0, 7, 4)
            humidite_sol_pct = st.slider("Humidité du sol (%)", 20.0, 100.0, 62.0)
            indice_eau_stagnante = st.slider("Indice d'eau stagnante (0-1)", 0.0, 1.0, 0.6)

        submitted = st.form_submit_button("Prédire le niveau de risque", type="primary", use_container_width=True)

    if submitted:
        # Coordonnées approximatives = moyenne nationale (non demandées à l'utilisateur)
        observation = {
            "mois": mois,
            "semaine": semaine,
            "region_administrative": region,
            "prefecture": prefecture,
            "zone_climatique": zone_climatique,
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
            "jour_annee": mois * 30,
        }

        result = predictor.predict_one(observation)
        niveau = result["niveau_risque_predit"]
        proba = result["probabilites"]

        st.divider()
        color = RISK_COLORS.get(niveau, "#666")
        st.markdown(
            f"### Niveau de risque prédit : "
            f"<span style='color:{color}'>**{niveau.upper()}**</span>",
            unsafe_allow_html=True,
        )

        proba_df = pd.DataFrame({"niveau_risque": list(proba.keys()), "probabilite": list(proba.values())})
        order = ["faible", "moyen", "eleve"]
        proba_df["niveau_risque"] = pd.Categorical(proba_df["niveau_risque"], categories=order, ordered=True)
        proba_df = proba_df.sort_values("niveau_risque")

        fig = px.bar(
            proba_df, x="niveau_risque", y="probabilite", color="niveau_risque",
            color_discrete_map=RISK_COLORS, text_auto=".1%",
            labels={"niveau_risque": "Niveau de risque", "probabilite": "Probabilité"},
        )
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        if niveau == "eleve":
            st.warning(
                "⚠️ Risque élevé : conditions favorables à la prolifération des vecteurs "
                "(eau stagnante, humidité, précipitations). Renforcer la sensibilisation et "
                "la disponibilité des traitements dans les centres de santé concernés."
            )


def page_exploration(df: pd.DataFrame) -> None:
    st.header("📊 Exploration des données")
    st.caption(f"{len(df):,} observations couvrant les {df['region_administrative'].nunique()} régions administratives de Guinée.")

    col1, col2 = st.columns([1, 1])
    with col1:
        risk_counts = df["niveau_risque"].value_counts().reindex(["faible", "moyen", "eleve"]).reset_index()
        risk_counts.columns = ["niveau_risque", "nombre"]
        fig = px.bar(
            risk_counts, x="niveau_risque", y="nombre", color="niveau_risque",
            color_discrete_map=RISK_COLORS, title="Répartition du niveau de risque",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_risk = pd.crosstab(df["region_administrative"], df["niveau_risque"], normalize="index")
        region_risk = region_risk.reindex(columns=["faible", "moyen", "eleve"], fill_value=0).reset_index()
        region_risk_melted = region_risk.melt(id_vars="region_administrative", var_name="niveau_risque", value_name="part")
        fig2 = px.bar(
            region_risk_melted, x="region_administrative", y="part", color="niveau_risque",
            color_discrete_map=RISK_COLORS, title="Part du risque par région",
            barmode="stack",
        )
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Relation météo / risque")
    weather_col = st.selectbox(
        "Variable météo",
        ["temperature_moyenne_c", "precipitation_7j_mm", "precipitation_30j_mm",
         "humidite_relative_pct", "humidite_sol_pct", "indice_eau_stagnante"],
    )
    fig3 = px.histogram(
        df, x=weather_col, color="niveau_risque", barmode="overlay",
        color_discrete_map=RISK_COLORS, histnorm="probability density", opacity=0.6,
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Voir un échantillon des données brutes"):
        st.dataframe(df.sample(min(200, len(df)), random_state=42), use_container_width=True)


def page_modele(metrics: dict) -> None:
    st.header("🧠 Performance du modèle")
    if not metrics:
        st.info("Aucune métrique disponible. Exécutez `python src/train.py` pour entraîner les modèles.")
        return

    st.caption(f"Modèle retenu : **{metrics['best_model']}** (meilleur F1 pondéré sur le jeu de test).")

    comp_df = pd.DataFrame(metrics["comparaison"]).T
    comp_df = comp_df.sort_values("f1_weighted", ascending=False)
    st.dataframe(comp_df.style.format("{:.3f}"), use_container_width=True)

    fig = px.bar(
        comp_df.reset_index().rename(columns={"index": "modele"}),
        x="modele", y=["accuracy", "precision_weighted", "rappel_weighted", "f1_weighted", "roc_auc_ovr_weighted"],
        barmode="group", title="Comparaison des modèles",
    )
    fig.update_layout(yaxis_range=[0, 1], legend_title="Métrique")
    st.plotly_chart(fig, use_container_width=True)

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


def main() -> None:
    st.title("🦟 Prédiction du risque de paludisme en Guinée")
    st.caption(
        "Projet data science — données synthétiques à but pédagogique. "
        "Ne remplace pas les données de surveillance épidémiologique officielles."
    )

    predictor = load_predictor()
    df = load_raw_data()
    metrics = load_metrics()

    tab1, tab2, tab3 = st.tabs(["Prédiction", "Exploration des données", "Performance du modèle"])
    with tab1:
        page_prediction(predictor)
    with tab2:
        page_exploration(df)
    with tab3:
        page_modele(metrics)

    st.sidebar.header("À propos")
    st.sidebar.write(
        "Cette application estime le niveau de risque de paludisme "
        "(faible / moyen / élevé) à partir de variables météorologiques "
        "et environnementales, pour aider à anticiper les périodes de "
        "forte demande dans les centres de santé."
    )
    st.sidebar.warning(
        "⚠️ Les données utilisées sont **synthétiques**. Une utilisation "
        "opérationnelle nécessiterait des données validées (OMS/HDX, "
        "services de santé, stations météo)."
    )


if __name__ == "__main__":
    main()
