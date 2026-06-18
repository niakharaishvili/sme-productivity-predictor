import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def generate_synthetic_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic SME dataset for prototyping.
    Replace with real HaMiZu or survey data when available.
    """
    np.random.seed(seed)

    sectors = ["Craft", "Trade", "Services", "Manufacturing", "Construction"]
    regions = ["Bremen", "Hamburg", "Berlin", "Bavaria", "NRW"]

    df = pd.DataFrame({
        "firm_id": range(1, n_samples + 1),
        "sector": np.random.choice(sectors, n_samples),
        "region": np.random.choice(regions, n_samples),
        "firm_size": np.random.randint(1, 250, n_samples),
        "years_in_operation": np.random.randint(1, 50, n_samples),
        "digitization_score": np.random.uniform(0, 10, n_samples),
        "innovation_index": np.random.uniform(0, 5, n_samples),
        "training_investment": np.random.uniform(0, 20000, n_samples),
    })

    # Simulate productivity as a weighted function of key variables + noise
    score = (
        0.35 * df["digitization_score"] +
        0.20 * df["innovation_index"] +
        0.15 * np.log1p(df["firm_size"]) +
        0.10 * np.log1p(df["training_investment"] / 1000) +
        np.random.normal(0, 1, n_samples)
    )

    df["productivity_label"] = pd.cut(
        score,
        bins=[-np.inf, score.quantile(0.33), score.quantile(0.66), np.inf],
        labels=["Low", "Medium", "High"]
    )

    return df


def load_raw_data(filename: str) -> pd.DataFrame:
    """Load raw data from the data/raw directory."""
    path = DATA_DIR / "raw" / filename
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save processed data to data/processed directory."""
    path = DATA_DIR / "processed" / filename
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")
