
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from evaluation import evaluate_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

dataset = load_breast_cancer(as_frame=True)
df = dataset.frame.copy()
df.to_csv(DATA_DIR / "breast_cancer_dataset.csv", index=False)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

test_data = X_test.copy()
test_data["target"] = y_test.values
test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "scaled": True,
        "file": "logistic_regression.pkl"
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=5, random_state=42),
        "scaled": False,
        "file": "decision_tree.pkl"
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(n_neighbors=5),
        "scaled": True,
        "file": "knn.pkl"
    },
    "Gaussian Naive Bayes": {
        "model": GaussianNB(),
        "scaled": False,
        "file": "naive_bayes.pkl"
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "scaled": False,
        "file": "random_forest.pkl"
    }
}

results = []

for name, cfg in models.items():
    model = cfg["model"]
    if cfg["scaled"]:
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        bundle_scaler = scaler
    else:
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        bundle_scaler = None

    joblib.dump(
        {
            "model": model,
            "scaler": bundle_scaler,
            "feature_names": list(X.columns)
        },
        MODEL_DIR / cfg["file"]
    )

    results.append({"ML Model Name": name, **metrics})

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

print("\nModel comparison:")
print(metrics_df.to_string(index=False))
print("\nSaved test_data.csv, model files and model_metrics.csv successfully.")
