"""
features.py — v3 feature engineering with group-aware interactions
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from data_loader import FEATURES, FEATURE_GROUPS

BINARY_COLS = [
    "h8_rd","h1_new_product","h5_new_process","h2_market_new",
    "c22b_website","l10_training",
    "b4_female_owner","b7a_female_manager","b3a_owner_is_manager",
    "r2_monitors_kpi","r4_has_targets","r8_perf_bonus","b8_quality_cert",
    "ge7_co2_monitor","ge8d_energy_mgmt","c43_solar",
    "e6_foreign_tech","k82a_has_loan","k21_audited","j36_etax",
]
ORDINAL_COLS = [
    "a4a_sector","a6a_size","a2_state","a3_city_size","e1_market_scope",
    "k30_finance_obs","l30b_workforce_obs","j30a_tax_obs",
    "j30f_corruption_obs","i30_crime_obs","d30a_transport_obs",
    "l30a_labor_reg_obs","l30c_hiring_cost_obs","l30d_dismissal_obs",
]
NUMERIC_COLS = [
    "l1","firm_age","employment_growth","sales_growth",
    "l9b_edu_pct","b7_mgr_experience",
    "d3c_export_pct","k3a_internal_funds_pct","k3bc_bank_debt_pct",
    "k33_epay_recv_pct","k38_epay_send_pct",
    "l11a_prod_training_pct","l12a_women_training_pct",
    "women_share","parttime_share",
    "j2_mgmt_time_govt","j35_tax_hours",
]


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Digital-innovation composite (5 binary signals)
    df["digital_innovation_score"] = (
        df["c22b_website"] + df["h8_rd"] +
        df["h1_new_product"] + df["h5_new_process"] + df["e6_foreign_tech"]
    )
    # Management quality index
    df["mgmt_quality_index"] = (
        df["r2_monitors_kpi"] + df["r4_has_targets"] +
        df["r8_perf_bonus"] + df["b8_quality_cert"]
    )
    # Green composite
    df["green_score"] = df["ge7_co2_monitor"] + df["ge8d_energy_mgmt"] + df["c43_solar"]

    # Obstacle burden index (all 6 obstacles summed)
    df["total_obstacle_burden"] = (
        df["k30_finance_obs"] + df["l30b_workforce_obs"] + df["j30a_tax_obs"] +
        df["j30f_corruption_obs"] + df["i30_crime_obs"] + df["d30a_transport_obs"]
    )
    # Labour regulation burden
    df["labour_reg_burden"] = (
        df["l30a_labor_reg_obs"] + df["l30c_hiring_cost_obs"] + df["l30d_dismissal_obs"]
    )
    # Human capital composite
    df["human_capital_index"] = (
        df["l10_training"] * df["l9b_edu_pct"] / 100 +
        df["l11a_prod_training_pct"] / 100
    )
    # E-payments digitisation (finance digitisation signal)
    df["epayment_adoption"] = (df["k33_epay_recv_pct"] + df["k38_epay_send_pct"]) / 2

    # Interaction: management × digital
    df["mgmt_x_digital"] = df["mgmt_quality_index"] * df["digital_innovation_score"]

    # Market orientation (export + market scope)
    df["market_orientation"] = df["e1_market_scope"] + (df["d3c_export_pct"] > 0).astype(int)

    return df

INTERACTION_COLS = [
    "digital_innovation_score","mgmt_quality_index","green_score",
    "total_obstacle_burden","labour_reg_burden","human_capital_index",
    "epayment_adoption","mgmt_x_digital","market_orientation",
]

ALL_FEATURES = FEATURES + INTERACTION_COLS


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
    ord_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                unknown_value=-1))])
    bin_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])

    # interaction cols are all numeric
    all_num = NUMERIC_COLS + INTERACTION_COLS

    return ColumnTransformer([
        ("num", num_pipe, all_num),
        ("ord", ord_pipe, ORDINAL_COLS),
        ("bin", bin_pipe, BINARY_COLS),
    ], remainder="drop")
