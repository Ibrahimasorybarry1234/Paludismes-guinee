"""
theme.py — Identité visuelle du dashboard Paludisme Guinée
------------------------------------------------------------
Placer ce fichier dans app/theme.py (à côté de app.py).
"""

import streamlit as st

# ---------------------------------------------------------------
# Tokens (cohérents avec .streamlit/config.toml)
# ---------------------------------------------------------------
FOREST_DEEP = "#0D241B"
FOREST = "#1B4332"
FOREST_SOFT = "#2D6A4F"
OCHRE = "#D4A017"
OCHRE_DARK = "#8a6a10"
RISK_HIGH = "#C62828"
SAND = "#F4EFE6"
INK = "#14201B"
INK_SOFT = "#4A5C53"

# Remplace le dict RISK_COLORS existant dans app.py — mêmes clés,
# nouvelles valeurs alignées sur l'identité (vert forêt / or / rouge risque).
RISK_COLORS = {"faible": FOREST_SOFT, "moyen": OCHRE, "eleve": RISK_HIGH}

_RISK_LABELS = {"faible": "Risque faible", "moyen": "Risque modéré", "eleve": "Risque élevé"}
_RISK_CLASS = {"faible": "low", "moyen": "mod", "eleve": "high"}


def apply_theme():
    """Injecte polices, badges, bandeau, et restyle les titres natifs Streamlit. Appeler une fois, juste après st.set_page_config()."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}

        /* Titres natifs (st.title / st.header / st.subheader) -> Fraunces */
        h1, h2, h3 {{
            font-family: 'Fraunces', serif !important;
            font-weight: 500 !important;
            color: {FOREST_DEEP} !important;
        }}

        /* Bandeau */
        .hero {{
            position: relative;
            background: linear-gradient(175deg, {FOREST} 0%, {FOREST_DEEP} 100%);
            color: {SAND};
            padding: 34px 38px 38px;
            border-radius: 18px;
            margin-bottom: 26px;
        }}
        .hero .hero-eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {OCHRE};
            margin-bottom: 8px;
        }}
        .hero h1 {{
            font-family: 'Fraunces', serif !important;
            font-weight: 500 !important;
            font-size: 32px;
            color: {SAND} !important;
            margin: 0 0 10px 0 !important;
        }}
        .hero p {{
            color: rgba(244,239,230,0.75);
            font-size: 14.5px;
            max-width: 640px;
            margin: 0;
        }}

        /* Badges de risque */
        .badge {{
            display: inline-flex; align-items: center; gap: 6px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px; font-weight: 500;
            padding: 5px 12px 5px 8px; border-radius: 100px;
        }}
        .badge .dot {{ width: 7px; height: 7px; border-radius: 50%; }}
        .badge.low  {{ background: rgba(45,106,79,0.12);  color: {FOREST}; }}
        .badge.low  .dot {{ background: {FOREST_SOFT}; }}
        .badge.mod  {{ background: rgba(212,160,23,0.16); color: {OCHRE_DARK}; }}
        .badge.mod  .dot {{ background: {OCHRE}; }}
        .badge.high {{ background: rgba(198,40,40,0.1);   color: {RISK_HIGH}; }}
        .badge.high .dot {{ background: {RISK_HIGH}; box-shadow: 0 0 0 3px rgba(198,40,40,0.16); }}

        /* Carte de résultat (moment clé de l'onglet Prédiction) */
        .result-card {{
            background: #fff;
            border-left: 4px solid var(--rc-color, {FOREST});
            border-radius: 4px 14px 14px 4px;
            padding: 20px 24px;
            margin: 4px 0 20px 0;
            box-shadow: 0 1px 3px rgba(20,32,27,0.06);
        }}
        .result-card .rc-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px; color: {INK_SOFT};
            text-transform: uppercase; letter-spacing: 0.08em;
            margin-bottom: 6px;
        }}
        .result-card .rc-value {{
            font-family: 'Fraunces', serif;
            font-weight: 600; font-size: 26px;
            color: var(--rc-color, {FOREST});
        }}

        /* Barre latérale */
        section[data-testid="stSidebar"] {{
            background: {SAND};
            border-right: 1px solid rgba(20,32,27,0.08);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_banner(eyebrow: str, title: str, subtitle: str):
    """Bandeau d'en-tête. À appeler une fois, en haut de main()."""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(niveau: str) -> str:
    """HTML d'un badge pill pour 'faible' / 'moyen' / 'eleve'."""
    css_class = _RISK_CLASS.get(niveau, "mod")
    label = _RISK_LABELS.get(niveau, niveau)
    return f'<span class="badge {css_class}"><span class="dot"></span>{label}</span>'


def result_card(niveau: str):
    """
    Remplace le bloc `### Niveau de risque prédit : <span style='color:...'>` de page_prediction.
    Usage : st.markdown(result_card(niveau), unsafe_allow_html=True)
    """
    color = RISK_COLORS.get(niveau, FOREST)
    label = _RISK_LABELS.get(niveau, niveau)
    return f"""
    <div class="result-card" style="--rc-color: {color};">
        <div class="rc-label">Niveau de risque prédit</div>
        <div class="rc-value">{label.upper()}</div>
    </div>
    """


def style_fig(fig):
    """Applique la typographie du projet aux graphiques Plotly. Appeler avant st.plotly_chart(fig)."""
    fig.update_layout(
        font_family="IBM Plex Sans",
        font_color=INK,
        title_font_family="Fraunces",
        title_font_size=18,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_font_family="IBM Plex Mono",
    )
    fig.update_xaxes(tickfont_family="IBM Plex Mono", gridcolor="rgba(20,32,27,0.06)")
    fig.update_yaxes(tickfont_family="IBM Plex Mono", gridcolor="rgba(20,32,27,0.06)")
    return fig
