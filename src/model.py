"""
model.py — training, cross-validation, SHAP, tuning
"""
import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GridSearchCV)
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def split(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_models(preprocessor) -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf",  LogisticRegression(max_iter=1000, random_state=42,
                                        class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf",  RandomForestClassifier(n_estimators=300, random_state=42,
                                             class_weight="balanced"))]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor),
            ("clf",  GradientBoostingClassifier(n_estimators=200,
                                                 learning_rate=0.05,
                                                 max_depth=4,
                                                 random_state=42))]),
    }


def cross_validate_all(models: dict, X, y, n_splits=5) -> pd.DataFrame:
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for name, pipe in models.items():
        s = cross_validate(pipe, X, y, cv=cv,
                           scoring=["accuracy","f1_macro","f1_weighted"],
                           return_train_score=True)
        rows.append({
            "Model":           name,
            "CV Accuracy":     f"{s['test_accuracy'].mean():.3f} +/- {s['test_accuracy'].std():.3f}",
            "CV F1 macro":     f"{s['test_f1_macro'].mean():.3f} +/- {s['test_f1_macro'].std():.3f}",
            "CV F1 weighted":  f"{s['test_f1_weighted'].mean():.3f} +/- {s['test_f1_weighted'].std():.3f}",
            "Train Accuracy":  f"{s['train_accuracy'].mean():.3f}",
            "_sort":           s["test_f1_macro"].mean(),
        })
    df = pd.DataFrame(rows).sort_values("_sort", ascending=False)
    return df.drop(columns="_sort").reset_index(drop=True)


def evaluate(pipeline, X_test, y_test, label_names=("Low","Medium","High")):
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_names))
    return {
        "report":           classification_report(y_test, y_pred,
                                                   target_names=label_names,
                                                   output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(label_names)),
        "accuracy":         accuracy_score(y_test, y_pred),
        "f1_macro":         f1_score(y_test, y_pred, average="macro"),
        "f1_weighted":      f1_score(y_test, y_pred, average="weighted"),
    }


def tune_random_forest(preprocessor, X_train, y_train):
    pipe = Pipeline([("prep", preprocessor),
                     ("clf",  RandomForestClassifier(random_state=42,
                                                      class_weight="balanced"))])
    grid = {
        "clf__n_estimators":     [200, 400],
        "clf__max_depth":        [None, 15, 25],
        "clf__min_samples_leaf": [1, 3],
        "clf__max_features":     ["sqrt", 0.5],
    }
    gs = GridSearchCV(pipe, grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV F1:  {gs.best_score_:.3f}")
    return gs.best_estimator_


def compute_shap(pipeline, X_test, feature_names):
    try:
        import shap
        prep = pipeline.named_steps["prep"]
        clf  = pipeline.named_steps["clf"]
        X_t  = prep.transform(X_test)
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_t)
        X_disp = pd.DataFrame(X_t, columns=feature_names[:X_t.shape[1]])
        return shap_values, X_disp, explainer
    except Exception as e:
        print(f"SHAP error: {e}")
        return None, None, None


def save_model(model, name):
    p = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, p)
    print(f"Saved -> {p}")

def load_model(name):
    return joblib.load(MODELS_DIR / f"{name}.pkl")
