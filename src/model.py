import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def split_data(X, y, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_baseline(X_train, y_train):
    """Logistic Regression baseline."""
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, tune: bool = False):
    """
    Train a Random Forest classifier.
    Set tune=True to run GridSearchCV for hyperparameter optimization.
    """
    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        rf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_}")
        return grid.best_estimator_
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        return model


def evaluate_model(model, X_test, y_test):
    """Print classification report and return confusion matrix."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return confusion_matrix(y_test, y_pred)


def save_model(model, name: str) -> None:
    """Persist a trained model to disk."""
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load_model(name: str):
    """Load a saved model from disk."""
    path = MODELS_DIR / f"{name}.pkl"
    return joblib.load(path)
