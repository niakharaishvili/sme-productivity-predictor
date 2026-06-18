"""
app.py : Interactive SME Productivity Predictor (WBES Germany 2025)
====================================================================
Run locally with:
    streamlit run app.py

Three tabs:
  1. Predict   : enter a firm profile, get a live productivity prediction
  2. Explore   : interactive charts of the dataset
  3. About     : project background and data source
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# make src importable
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from data_loader import load_data, TARGET, FEATURE_GROUPS
from features import add_interactions, build_preprocessor, ALL_FEATURES
from model import split, build_models, load_model, save_model

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SME Productivity Predictor : WBES Germany",
    page_icon="📊",
    layout="wide",
)

PALETTE = {"Low": "#d73027", "Medium": "#fee090", "High": "#1a9850"}
ORDER   = ["Low", "Medium", "High"]


# ── cached loaders ─────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_data()
    return add_interactions(df)

@st.cache_resource
def get_model(_df):
    """Load saved model, or train one if it does not exist."""
    try:
        return load_model("wbes_germany_v3_rf")
    except Exception:
        X = _df[[c for c in ALL_FEATURES if c in _df.columns]]
        y = _df[TARGET]
        X_tr, _, y_tr, _ = split(X, y)
        prep = build_preprocessor()
        rf = build_models(prep)["Random Forest"]
        rf.fit(X_tr, y_tr)
        save_model(rf, "wbes_germany_v3_rf")
        return rf


df    = get_data()
model = get_model(df)
feature_cols = [c for c in ALL_FEATURES if c in df.columns]


# ── header ─────────────────────────────────────────────────────────────────
st.title("📊 SME Productivity Predictor")
st.markdown(
    "Predicting German SME productivity from the **World Bank Enterprise "
    "Survey Germany 2025** (`DEU_2025_WBES_v01_M`). "
    "53 variables across 7 themes : digital adoption, management quality, "
    "green practices, finance, labour, market orientation and regulation."
)

tab_predict, tab_explore, tab_about = st.tabs(
    ["🔮 Predict", "📈 Explore Data", "ℹ️ About"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 : PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Enter a firm profile")
    st.caption("Adjust the controls, then read the live prediction on the right.")

    left, right = st.columns([3, 2])

    with left:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Firm basics**")
            l1 = st.slider("Employees", 5, 1000, 50)
            firm_age = st.slider("Firm age (years)", 1, 65, 15)
            a6a_size = st.selectbox("Size class",
                                    [1, 2, 3, 4],
                                    format_func=lambda x: {1: "Small (5-19)",
                                                           2: "Medium (20-99)",
                                                           3: "Large (100-249)",
                                                           4: "Very Large (250+)"}[x])
            a4a_sector = st.selectbox("Sector code", list(range(1, 10)))

        with c2:
            st.markdown("**Digital & innovation**")
            c22b_website   = st.checkbox("Has website", True)
            h8_rd          = st.checkbox("Spends on R&D", False)
            h1_new_product = st.checkbox("New product (3 yrs)", True)
            h5_new_process = st.checkbox("New process (3 yrs)", False)
            e6_foreign_tech = st.checkbox("Uses foreign tech", False)
            b8_quality_cert = st.checkbox("Quality certification", False)

        with c3:
            st.markdown("**Management & people**")
            r2_monitors_kpi = st.checkbox("Monitors KPIs", True)
            r4_has_targets  = st.checkbox("Has targets", True)
            r8_perf_bonus   = st.checkbox("Performance bonuses", False)
            l10_training    = st.checkbox("Formal training", True)
            l9b_edu_pct     = st.slider("% workers w/ high school", 0, 100, 70)
            b7a_female_manager = st.checkbox("Female top manager", False)

        st.markdown("**Obstacles** (0 = none, 4 = very severe)")
        o1, o2, o3 = st.columns(3)
        with o1:
            k30 = st.slider("Finance", 0, 4, 1)
        with o2:
            l30b = st.slider("Workforce skills", 0, 4, 1)
        with o3:
            j30a = st.slider("Tax rates", 0, 4, 2)

    # build a single-row feature frame using dataset medians as defaults
    profile = df[feature_cols].median(numeric_only=True).to_dict()
    profile.update({
        "l1": l1, "firm_age": firm_age, "a6a_size": a6a_size,
        "a4a_sector": a4a_sector,
        "c22b_website": int(c22b_website), "h8_rd": int(h8_rd),
        "h1_new_product": int(h1_new_product),
        "h5_new_process": int(h5_new_process),
        "e6_foreign_tech": int(e6_foreign_tech),
        "b8_quality_cert": int(b8_quality_cert),
        "r2_monitors_kpi": int(r2_monitors_kpi),
        "r4_has_targets": int(r4_has_targets),
        "r8_perf_bonus": int(r8_perf_bonus),
        "l10_training": int(l10_training),
        "l9b_edu_pct": l9b_edu_pct,
        "b7a_female_manager": int(b7a_female_manager),
        "k30_finance_obs": k30, "l30b_workforce_obs": l30b,
        "j30a_tax_obs": j30a,
        # recompute the composite scores that depend on inputs
        "digital_innovation_score": int(c22b_website) + int(h8_rd)
                                   + int(h1_new_product) + int(h5_new_process)
                                   + int(e6_foreign_tech),
        "mgmt_quality_index": int(r2_monitors_kpi) + int(r4_has_targets)
                             + int(r8_perf_bonus) + int(b8_quality_cert),
    })
    X_one = pd.DataFrame([profile])[feature_cols]

    with right:
        st.markdown("### Prediction")
        pred   = model.predict(X_one)[0]
        proba  = model.predict_proba(X_one)[0]
        classes = list(model.classes_)

        color = PALETTE.get(pred, "#333")
        st.markdown(
            f"<div style='padding:1.2rem;border-radius:12px;background:{color};"
            f"text-align:center;color:white;font-size:1.6rem;font-weight:700'>"
            f"{pred} Productivity</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Confidence")
        prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
        prob_df = prob_df.set_index("Class").reindex(ORDER).fillna(0)
        for cls in ORDER:
            p = float(prob_df.loc[cls, "Probability"])
            st.markdown(f"**{cls}**")
            st.progress(p, text=f"{p*100:.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 : EXPLORE
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("Explore the dataset")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Firms in dataset", len(df))
    with c2:
        st.metric("Variables", df.shape[1])

    st.markdown("#### Productivity distribution")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    counts = df[TARGET].value_counts().reindex(ORDER)
    ax.bar(ORDER, counts, color=[PALETTE[o] for o in ORDER], edgecolor="white")
    ax.set_ylabel("Firms")
    sns.despine(ax=ax)
    st.pyplot(fig)

    st.markdown("#### Adoption rate by productivity class")
    metric = st.selectbox(
        "Choose a metric",
        ["c22b_website", "h8_rd", "l10_training", "b8_quality_cert",
         "r2_monitors_kpi", "ge7_co2_monitor"],
        format_func=lambda c: {
            "c22b_website": "Has website",
            "h8_rd": "R&D spend",
            "l10_training": "Formal training",
            "b8_quality_cert": "Quality certification",
            "r2_monitors_kpi": "Monitors KPIs",
            "ge7_co2_monitor": "Monitors CO2",
        }.get(c, c),
    )
    rates = df.groupby(TARGET)[metric].mean().reindex(ORDER) * 100
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.bar(ORDER, rates, color=[PALETTE[o] for o in ORDER], edgecolor="white")
    ax2.set_ylabel("% of firms")
    ax2.set_ylim(0, 100)
    sns.despine(ax=ax2)
    st.pyplot(fig2)

    with st.expander("Show raw data sample"):
        st.dataframe(df.head(50))


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 : ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("About this project")
    st.markdown("""
This dashboard sits on top of a machine-learning pipeline that predicts the
productivity class of a German SME (Low / Medium / High, based on capacity
utilisation) from its characteristics.

**Data**  : World Bank Enterprise Survey Germany 2025 (`DEU_2025_WBES_v01_M`).
The app ships with a synthetic dataset that mirrors the real survey's variables
and distributions; dropping the real CSV into `data/raw/` switches the whole
pipeline to real data with no code changes.

**Model** : Random Forest (compared in the notebooks against Logistic Regression
and Gradient Boosting via 5-fold cross-validation), with SHAP explainability.

**Themes covered** : digital adoption, management quality, green practices,
finance, labour, market orientation, and regulatory burden.

Built by **Nia Kharaishvili** : inspired by the BMBF-funded HaMiZu project on
SME digital transformation.
""")
    st.markdown("**Feature groups**")
    for group, cols in FEATURE_GROUPS.items():
        st.markdown(f"- **{group}** ({len(cols)} variables)")
