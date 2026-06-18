"""
visualize.py: v3 rich visualisations for WBES Germany project
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REPORTS.mkdir(exist_ok=True)

PALETTE = {"Low":"#d73027","Medium":"#fee090","High":"#1a9850"}
ORDER   = ["Low","Medium","High"]
FT, FL  = 14, 11


def _save(fig, name, save):
    if save:
        p = REPORTS / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")


def plot_target_distribution(df, target="productivity_class", save=False):
    fig, ax = plt.subplots(figsize=(7,4))
    counts = df[target].value_counts().reindex(ORDER)
    bars = ax.bar(ORDER, counts, color=[PALETTE[o] for o in ORDER], edgecolor="white", linewidth=1.2)
    for b,v in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+3, str(int(v)), ha="center", va="bottom")
    ax.set_title("Firm Productivity Distribution: WBES Germany 2025", fontsize=FT)
    ax.set_xlabel("Productivity class (capacity utilisation tercile)")
    ax.set_ylabel("Number of firms")
    sns.despine(ax=ax); plt.tight_layout(); _save(fig,"01_target",save); plt.show()


def plot_theme_radar(df, target="productivity_class", save=False):
    """Radar chart: average score per theme for each productivity class."""
    themes = {
        "Digital\nInnovation": "digital_innovation_score",
        "Management\nQuality":  "mgmt_quality_index",
        "Green\nPractices":     "green_score",
        "E-Payment\nAdoption":  "epayment_adoption",
        "Human\nCapital":       "human_capital_index",
        "Market\nOrientation":  "market_orientation",
    }
    labels = list(themes.keys())
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    for prod in ORDER:
        sub = df[df[target]==prod]
        vals = [sub[col].mean() for col in themes.values()]
        # normalise each metric 0-1
        all_vals = [df[col].mean() for col in themes.values()]
        max_v = [max(df[col].max(), 1e-9) for col in themes.values()]
        norm = [v/m for v,m in zip(vals, max_v)] + [vals[0]/max_v[0]]
        ax.plot(angles, norm, "o-", linewidth=2, label=prod, color=PALETTE[prod])
        ax.fill(angles, norm, alpha=0.08, color=PALETTE[prod])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=FL)
    ax.set_title("Capability Profile by Productivity Class", fontsize=FT, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    plt.tight_layout(); _save(fig,"02_radar",save); plt.show()


def plot_digital_adoption(df, target="productivity_class", save=False):
    cols = {"Website (c22b)":"c22b_website","New Product (h1)":"h1_new_product",
            "New Process (h5)":"h5_new_process","R&D (h8)":"h8_rd",
            "Foreign Tech (e6)":"e6_foreign_tech","Training (l10)":"l10_training",
            "Quality Cert (b8)":"b8_quality_cert"}
    x = np.arange(len(cols)); w = 0.25
    fig, ax = plt.subplots(figsize=(13,5))
    for i,prod in enumerate(ORDER):
        vals = [df[df[target]==prod][c].mean()*100 for c in cols.values()]
        ax.bar(x+i*w, vals, w, label=prod, color=PALETTE[prod], edgecolor="white")
    ax.set_xticks(x+w); ax.set_xticklabels(list(cols.keys()), fontsize=FL-1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Digital & Innovation Adoption by Productivity Class", fontsize=FT)
    ax.legend(title="Productivity"); sns.despine(ax=ax)
    plt.tight_layout(); _save(fig,"03_digital_adoption",save); plt.show()


def plot_obstacle_heatmap(df, target="productivity_class", save=False):
    obs = {"Finance (k30)":"k30_finance_obs","Workforce (l30b)":"l30b_workforce_obs",
           "Tax Rates (j30a)":"j30a_tax_obs","Corruption (j30f)":"j30f_corruption_obs",
           "Crime (i30)":"i30_crime_obs","Transport (d30a)":"d30a_transport_obs",
           "Labour Reg (l30a)":"l30a_labor_reg_obs","Hiring Cost (l30c)":"l30c_hiring_cost_obs",
           "Dismissal (l30d)":"l30d_dismissal_obs"}
    heat = pd.DataFrame({l: df.groupby(target)[c].mean().reindex(ORDER)
                         for l,c in obs.items()})
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(heat.T, annot=True, fmt=".2f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax, cbar_kws={"label":"Mean severity (0-4)"})
    ax.set_title("Full Business Obstacle Profile by Productivity Class", fontsize=FT)
    ax.set_xlabel("Productivity Class"); ax.set_ylabel("")
    plt.tight_layout(); _save(fig,"04_obstacle_heatmap",save); plt.show()


def plot_management_profile(df, target="productivity_class", save=False):
    mgmt_cols = {"Monitors KPIs\n(r2)":"r2_monitors_kpi","Has Targets\n(r4)":"r4_has_targets",
                 "Perf Bonuses\n(r8)":"r8_perf_bonus","Quality Cert\n(b8)":"b8_quality_cert",
                 "Owner=Manager\n(b3a)":"b3a_owner_is_manager"}
    x = np.arange(len(mgmt_cols)); w = 0.25
    fig, ax = plt.subplots(figsize=(11,5))
    for i,prod in enumerate(ORDER):
        vals = [df[df[target]==prod][c].mean()*100 for c in mgmt_cols.values()]
        ax.bar(x+i*w, vals, w, label=prod, color=PALETTE[prod], edgecolor="white")
    ax.set_xticks(x+w); ax.set_xticklabels(list(mgmt_cols.keys()))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Management Quality Practices by Productivity Class", fontsize=FT)
    ax.legend(title="Productivity"); sns.despine(ax=ax)
    plt.tight_layout(); _save(fig,"05_management",save); plt.show()


def plot_green_finance(df, target="productivity_class", save=False):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    # green
    gcols = {"CO2 Monitor\n(ge7)":"ge7_co2_monitor","Energy Mgmt\n(ge8d)":"ge8d_energy_mgmt",
             "Solar Panels\n(c43)":"c43_solar"}
    x = np.arange(len(gcols)); w = 0.25
    for i,prod in enumerate(ORDER):
        vals = [df[df[target]==prod][c].mean()*100 for c in gcols.values()]
        axes[0].bar(x+i*w, vals, w, label=prod, color=PALETTE[prod], edgecolor="white")
    axes[0].set_xticks(x+w); axes[0].set_xticklabels(list(gcols.keys()))
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[0].set_title("Green Practices Adoption", fontsize=FT); axes[0].legend(title="Productivity")
    sns.despine(ax=axes[0])
    # e-payments
    for prod in ORDER:
        sub = df[df[target]==prod]
        axes[1].bar([prod+"\nReceive", prod+"\nSend"],
                    [sub["k33_epay_recv_pct"].mean(), sub["k38_epay_send_pct"].mean()],
                    color=PALETTE[prod], edgecolor="white", alpha=0.85)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[1].set_title("E-Payment Adoption (% transactions)", fontsize=FT)
    axes[1].set_ylabel("% transactions via e-payment"); sns.despine(ax=axes[1])
    plt.tight_layout(); _save(fig,"06_green_finance",save); plt.show()


def plot_german_state_map(df, target="productivity_class", save=False):
    state_names = {1:"Baden-Wuerttemberg",2:"Bayern",3:"Berlin",4:"Brandenburg",
                   5:"Bremen",6:"Hamburg",7:"Hessen",8:"Mecklenburg-Vorpommern",
                   9:"Niedersachsen",10:"NRW",11:"Rheinland-Pfalz",12:"Saarland",
                   13:"Sachsen",14:"Sachsen-Anhalt",15:"Schleswig-Holstein",16:"Thueringen"}
    df2 = df.copy()
    df2["state_name"] = df2["a2_state"].map(state_names)
    pct = (df2[df2[target]=="High"].groupby("state_name").size() /
           df2.groupby("state_name").size() * 100).fillna(0).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9,6))
    pct.plot.barh(ax=ax, color="#1a9850", edgecolor="white")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("% High-Productivity Firms by German Federal State", fontsize=FT)
    ax.set_xlabel("% of firms in state classified as High Productivity")
    sns.despine(ax=ax); plt.tight_layout(); _save(fig,"07_states",save); plt.show()


def plot_cv_results(cv_df, save=False):
    models = cv_df["Model"].tolist()
    acc = [float(v.split(" +/-")[0]) for v in cv_df["CV Accuracy"]]
    f1  = [float(v.split(" +/-")[0]) for v in cv_df["CV F1 macro"]]
    x   = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(x-0.2, acc, 0.35, label="Accuracy", color="#4393c3")
    ax.bar(x+0.2, f1,  0.35, label="F1 macro",  color="#d6604d")
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0,1)
    ax.set_title("5-Fold Cross-Validation Comparison", fontsize=FT)
    ax.legend(); sns.despine(ax=ax); plt.tight_layout()
    _save(fig,"08_cv_results",save); plt.show()


def plot_confusion_matrix(cm, labels=ORDER, save=False):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix: Best Model (Test Set)", fontsize=FT)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout(); _save(fig,"09_confusion_matrix",save); plt.show()


def plot_feature_importance(pipeline, feature_names, top_n=15, save=False):
    imp = pipeline.named_steps["clf"].feature_importances_
    n   = min(top_n, len(imp), len(feature_names))
    idx = np.argsort(imp)[::-1][:n]
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=imp[idx], y=[feature_names[i] for i in idx], palette="viridis", ax=ax)
    ax.set_title(f"Top {n} Feature Importances: Random Forest", fontsize=FT)
    ax.set_xlabel("Mean decrease in impurity")
    sns.despine(ax=ax); plt.tight_layout()
    _save(fig,"10_feature_importance",save); plt.show()


def plot_shap_summary(shap_values, X_disp, class_idx=2, class_name="High Productivity", save=False):
    try:
        import shap
        shap.summary_plot(shap_values[class_idx], X_disp,
                          plot_type="dot", show=False,
                          title=f"SHAP: {class_name}")
        plt.tight_layout()
        _save(plt.gcf(),"11_shap_summary",save); plt.show()
    except Exception as e:
        print(f"SHAP plot skipped: {e}")
