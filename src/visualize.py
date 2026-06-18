import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def plot_productivity_distribution(df, save: bool = False):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["Low", "Medium", "High"]
    sns.countplot(data=df, x="productivity_label", order=order, palette="Blues_d", ax=ax)
    ax.set_title("SME Productivity Label Distribution")
    ax.set_xlabel("Productivity Level")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "productivity_distribution.png", dpi=150)
    plt.show()


def plot_digitization_vs_productivity(df, save: bool = False):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Low", "Medium", "High"]
    sns.boxplot(data=df, x="productivity_label", y="digitization_score",
                order=order, palette="Set2", ax=ax)
    ax.set_title("Digitization Score by Productivity Level")
    ax.set_xlabel("Productivity Level")
    ax.set_ylabel("Digitization Score (0–10)")
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "digitization_vs_productivity.png", dpi=150)
    plt.show()


def plot_sector_breakdown(df, save: bool = False):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["Low", "Medium", "High"]
    sns.countplot(data=df, x="sector", hue="productivity_label",
                  hue_order=order, palette="Set1", ax=ax)
    ax.set_title("Productivity Distribution by Sector")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Count")
    plt.legend(title="Productivity")
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "sector_breakdown.png", dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 10, save: bool = False):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances[indices],
                y=[feature_names[i] for i in indices],
                palette="viridis", ax=ax)
    ax.set_title("Top Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.show()


def plot_confusion_matrix(cm, labels, save: bool = False):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.show()
