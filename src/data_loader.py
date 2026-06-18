"""
data_loader.py  —  WBES Germany 2025  (v3)
===========================================
53 variables from codebook DEU_2025_WBES_v01_M.xml across 7 themes:
  Core · Management Quality · Green · Market · Finance · Labour · Regulation

Drop real CSV at data/raw/DEU_2025_WBES_v01_M.csv — loader switches automatically.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REAL_CSV  = "DEU_2025_WBES_v01_M.csv"

FEATURES_CORE = [
    "l1","firm_age","employment_growth","sales_growth",
    "h8_rd","h1_new_product","h5_new_process","h2_market_new",
    "c22b_website","l10_training","l9b_edu_pct",
    "b4_female_owner","b7a_female_manager","b7_mgr_experience",
    "a4a_sector","a6a_size","a2_state","a3_city_size",
]
FEATURES_MGMT = [
    "r2_monitors_kpi","r4_has_targets","r8_perf_bonus",
    "b8_quality_cert","b3a_owner_is_manager",
]
FEATURES_GREEN  = ["ge7_co2_monitor","ge8d_energy_mgmt","c43_solar"]
FEATURES_MARKET = ["e1_market_scope","e6_foreign_tech","d3c_export_pct"]
FEATURES_FINANCE = [
    "k3a_internal_funds_pct","k3bc_bank_debt_pct",
    "k82a_has_loan","k21_audited","k33_epay_recv_pct","k38_epay_send_pct",
]
FEATURES_LABOUR = [
    "l11a_prod_training_pct","l12a_women_training_pct",
    "women_share","parttime_share",
]
FEATURES_REGULATION = [
    "j2_mgmt_time_govt","j35_tax_hours","j36_etax",
    "l30a_labor_reg_obs","l30c_hiring_cost_obs","l30d_dismissal_obs",
]
FEATURES_OBSTACLES = [
    "k30_finance_obs","l30b_workforce_obs","j30a_tax_obs",
    "j30f_corruption_obs","i30_crime_obs","d30a_transport_obs",
]

FEATURES = (FEATURES_CORE + FEATURES_MGMT + FEATURES_GREEN +
            FEATURES_MARKET + FEATURES_FINANCE + FEATURES_LABOUR +
            FEATURES_REGULATION + FEATURES_OBSTACLES)
TARGET = "productivity_class"

FEATURE_GROUPS = {
    "Core":        FEATURES_CORE,
    "Management":  FEATURES_MGMT,
    "Green":       FEATURES_GREEN,
    "Market":      FEATURES_MARKET,
    "Finance":     FEATURES_FINANCE,
    "Labour":      FEATURES_LABOUR,
    "Regulation":  FEATURES_REGULATION,
    "Obstacles":   FEATURES_OBSTACLES,
}


def load_data() -> pd.DataFrame:
    p = DATA_RAW / REAL_CSV
    if p.exists():
        print(f"Loading real WBES CSV: {p}")
        return _load_real(p)
    print("Synthetic WBES-mirrored data (53 variables, 700 firms).")
    print(f"Drop real CSV at:  {DATA_RAW / REAL_CSV}")
    return _generate_synthetic()


def _neg_to_nan(s):
    return pd.to_numeric(s, errors="coerce").replace([-9,-8,-7,-6,-5,-4], np.nan)

def _load_real(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    yn  = lambda c: (_neg_to_nan(raw.get(c)) == 1).astype(float)
    num = lambda c:  _neg_to_nan(raw.get(c))

    df = pd.DataFrame({
        "l1": num("l1"), "l2": num("l2"), "b5": num("b5"),
        "d2": num("d2"), "n3": num("n3"),
        "h8_rd": yn("h8"), "h1_new_product": yn("h1"),
        "h5_new_process": yn("h5"), "h2_market_new": yn("h2"),
        "c22b_website": yn("c22b"), "l10_training": yn("l10"),
        "l9b_edu_pct": num("l9b"),
        "b4_female_owner": yn("b4"), "b7a_female_manager": yn("b7a"),
        "b7_mgr_experience": num("b7"),
        "a4a_sector": num("a4a"), "a6a_size": num("a6a"),
        "a2_state": num("a2"), "a3_city_size": num("a3"),
        "r2_monitors_kpi": yn("r2"), "r4_has_targets": yn("r4"),
        "r8_perf_bonus": yn("r8"), "b8_quality_cert": yn("b8"),
        "b3a_owner_is_manager": yn("b3a"),
        "ge7_co2_monitor": yn("ge7"), "ge8d_energy_mgmt": yn("ge8d"),
        "c43_solar": yn("c43"),
        "e1_market_scope": num("e1"), "e6_foreign_tech": yn("e6"),
        "d3c_export_pct": num("d3c"),
        "k3a_internal_funds_pct": num("k3a"), "k3bc_bank_debt_pct": num("k3bc"),
        "k82a_has_loan": yn("k82a"), "k21_audited": yn("k21"),
        "k33_epay_recv_pct": num("k33"), "k38_epay_send_pct": num("k38"),
        "l5_women": num("l5"), "l1a_parttime": num("l1a"),
        "l11a_prod_training_pct": num("l11a"),
        "l12a_women_training_pct": num("l12a"),
        "j2_mgmt_time_govt": num("j2"), "j35_tax_hours": num("j35"),
        "j36_etax": (_neg_to_nan(raw.get("j36")) == 1).astype(float),
        "l30a_labor_reg_obs": num("l30a"), "l30c_hiring_cost_obs": num("l30c"),
        "l30d_dismissal_obs": num("l30d"),
        "k30_finance_obs": num("k30"), "l30b_workforce_obs": num("l30b"),
        "j30a_tax_obs": num("j30a"), "j30f_corruption_obs": num("j30f"),
        "i30_crime_obs": num("i30"), "d30a_transport_obs": num("d30a"),
        "f1_capacity": num("f1"),
    })
    df["firm_age"]          = 2025 - df["b5"]
    df["employment_growth"] = (df["l1"] - df["l2"]) / df["l2"].replace(0, np.nan)
    df["sales_growth"]      = (df["d2"] - df["n3"]) / df["n3"].replace(0, np.nan)
    df["women_share"]       = df["l5_women"] / df["l1"].replace(0, np.nan)
    df["parttime_share"]    = df["l1a_parttime"] / df["l1"].replace(0, np.nan)
    df[TARGET] = pd.qcut(df["f1_capacity"], q=3, labels=["Low","Medium","High"])
    return _clean(df)


def _generate_synthetic(n=700, seed=42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    size = rng.choice([1,2,3,4], n, p=[0.45,0.30,0.15,0.10])
    sect = rng.integers(1, 10, n)
    state= rng.integers(1, 17, n)
    city = rng.choice([2,3,4,5], n, p=[0.20,0.25,0.30,0.25])
    age  = rng.integers(2, 65, n)
    emp  = np.where(size==1, rng.integers(5,20,n),
           np.where(size==2, rng.integers(20,100,n),
           np.where(size==3, rng.integers(100,250,n),
                             rng.integers(250,1000,n))))
    emp_3y  = np.maximum(1, emp + rng.integers(-10,15,n))
    grow    = rng.normal(0.05, 0.15, n)
    web     = rng.binomial(1,0.72,n);  rd  = rng.binomial(1,0.38,n)
    np_     = rng.binomial(1,0.50,n);  npc = rng.binomial(1,0.44,n)
    mktN    = rng.binomial(1,0.30,n);  trn = rng.binomial(1,0.60,n)
    edu     = rng.uniform(40,100,n)
    fo      = rng.binomial(1,0.22,n);  fm  = rng.binomial(1,0.18,n)
    mgr_exp = rng.integers(1,35,n)
    kpi     = rng.binomial(1,0.65,n);  tgt = rng.binomial(1,0.60,n)
    bon     = rng.binomial(1,0.55,n);  cert= rng.binomial(1,0.35,n)
    ownm    = rng.binomial(1,0.48,n)
    co2     = rng.binomial(1,0.40,n);  enrg= rng.binomial(1,0.55,n)
    sol     = rng.binomial(1,0.25,n)
    mkt     = rng.choice([1,2,3],n,p=[0.35,0.40,0.25])
    ftech   = rng.binomial(1,0.28,n)
    exp_pct = np.where(mkt==3, rng.uniform(10,60,n), rng.uniform(0,10,n))
    int_f   = rng.uniform(40,90,n);   bank_d = rng.uniform(0,30,n)
    loan    = rng.binomial(1,0.35,n); aud   = rng.binomial(1,0.58,n)
    ep_r    = rng.uniform(30,95,n);   ep_s  = rng.uniform(50,98,n)
    wsh     = rng.uniform(0.10,0.55,n)
    pts     = rng.uniform(0.02,0.20,n)
    tp      = rng.uniform(20,80,n);   tw = rng.uniform(20,75,n)
    mgt     = rng.uniform(1,25,n);    txh= rng.uniform(10,150,n)
    etx     = rng.binomial(1,0.82,n)
    lro     = rng.integers(0,5,n);    hco= rng.integers(0,5,n)
    dmo     = rng.integers(0,5,n)
    fobs    = rng.integers(0,5,n);    wobs= rng.integers(0,5,n)
    tobs    = rng.integers(0,5,n);    cobs= rng.integers(0,5,n)
    crobs   = rng.integers(0,5,n);    trobs=rng.integers(0,5,n)

    cap = np.clip(
        55
        + 7*web + 5*rd + 4*np_ + 3*npc + 5*trn
        + 6*kpi + 5*tgt + 4*bon + 4*cert
        + 3*co2 + 3*enrg
        + 4*(mkt-1) + 5*ftech + 0.08*exp_pct
        + 0.05*int_f - 0.03*bank_d
        + 0.08*edu + 0.05*mgr_exp
        - 3*fobs - 2*wobs - tobs - cobs
        - 0.05*mgt - 0.02*txh
        + rng.normal(0, 8, n),
        5, 100)

    df = pd.DataFrame({
        "l1": emp, "firm_age": age,
        "employment_growth": (emp-emp_3y)/np.maximum(emp_3y,1),
        "sales_growth": grow,
        "h8_rd": rd, "h1_new_product": np_, "h5_new_process": npc,
        "h2_market_new": mktN, "c22b_website": web, "l10_training": trn,
        "l9b_edu_pct": edu, "b4_female_owner": fo, "b7a_female_manager": fm,
        "b7_mgr_experience": mgr_exp, "a4a_sector": sect,
        "a6a_size": size, "a2_state": state, "a3_city_size": city,
        "r2_monitors_kpi": kpi, "r4_has_targets": tgt, "r8_perf_bonus": bon,
        "b8_quality_cert": cert, "b3a_owner_is_manager": ownm,
        "ge7_co2_monitor": co2, "ge8d_energy_mgmt": enrg, "c43_solar": sol,
        "e1_market_scope": mkt, "e6_foreign_tech": ftech, "d3c_export_pct": exp_pct,
        "k3a_internal_funds_pct": int_f, "k3bc_bank_debt_pct": bank_d,
        "k82a_has_loan": loan, "k21_audited": aud,
        "k33_epay_recv_pct": ep_r, "k38_epay_send_pct": ep_s,
        "l11a_prod_training_pct": tp, "l12a_women_training_pct": tw,
        "women_share": wsh, "parttime_share": pts,
        "j2_mgmt_time_govt": mgt, "j35_tax_hours": txh, "j36_etax": etx,
        "l30a_labor_reg_obs": lro, "l30c_hiring_cost_obs": hco,
        "l30d_dismissal_obs": dmo,
        "k30_finance_obs": fobs, "l30b_workforce_obs": wobs,
        "j30a_tax_obs": tobs, "j30f_corruption_obs": cobs,
        "i30_crime_obs": crobs, "d30a_transport_obs": trobs,
        "f1_capacity": cap,
    })
    df[TARGET] = pd.qcut(df["f1_capacity"], q=3, labels=["Low","Medium","High"])
    return _clean(df)


def _clean(df):
    df = df.dropna(subset=[TARGET])
    num = df.select_dtypes(include=np.number).columns
    df[num] = df[num].fillna(df[num].median())
    return df.reset_index(drop=True)

def save_processed(df, name="wbes_germany_v3_processed.csv"):
    p = DATA_PROC / name
    df.to_csv(p, index=False)
    print(f"Saved -> {p}")
