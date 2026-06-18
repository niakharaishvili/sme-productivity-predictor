import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Label encode categorical columns."""
    df = df.copy()
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that capture interaction effects."""
    df = df.copy()
    # Interaction: digitization amplified by firm size
    df["size_x_digital"] = df["firm_size"] * df["digitization_score"]
    # Diminishing returns on firm size
    df["sqrt_firm_size"] = np.sqrt(df["firm_size"])
    # Training efficiency per employee
    df["training_per_employee"] = df["training_investment"] / (df["firm_size"] + 1)
    # Combined innovation-digitization signal
    df["innovation_x_digital"] = df["innovation_index"] * df["digitization_score"]
    return df


def get_feature_matrix(df: pd.DataFrame, target_col: str = "productivity_label"):
    """Return X (features) and y (target) ready for sklearn."""
    cat_cols = ["sector", "region"]
    df_enc = encode_categoricals(df, cat_cols)
    df_feat = engineer_features(df_enc)

    drop_cols = [target_col, "firm_id"]
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])
    y = df_feat[target_col]
    return X, y
