
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

st.set_page_config(
    page_title="Breast Cancer Classification - ML Assignment 2",
    page_icon="🧠",
    layout="wide"
)

st.title("Breast Cancer Diagnostic Classification")
st.caption("Machine Learning Assignment 2 | Five Classification Models")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Project Overview", "Dataset Explorer", "Model Evaluation", "Test Data Prediction"]
)

@st.cache_data
def load_metrics():
    return pd.read_csv(BASE_DIR / "results" / "model_metrics.csv")

@st.cache_data
def load_test_data():
    return pd.read_csv(BASE_DIR / "test_data.csv")

@st.cache_resource
def load_model(model_name):
    return joblib.load(BASE_DIR / "models" / MODEL_FILES[model_name])


def get_positive_class_scores(model, X_input):
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_input)[:, 1], "probability", None
    except Exception as exc:
        probability_error = str(exc)
    else:
        probability_error = None

    try:
        if hasattr(model, "decision_function"):
            return model.decision_function(X_input), "score", probability_error
    except Exception as exc:
        score_error = str(exc)
        if probability_error:
            return None, None, f"predict_proba failed: {probability_error}; decision_function failed: {score_error}"
        return None, None, f"decision_function failed: {score_error}"

    if probability_error:
        return None, None, f"predict_proba failed: {probability_error}"
    return None, None, None

metrics_df = load_metrics()
test_df = load_test_data()

if page == "Project Overview":
    st.subheader("Problem Statement")
    st.write(
        "The objective of this project is to classify breast tumour records as "
        "malignant or benign using five machine learning classification models "
        "and compare their performance using multiple evaluation metrics."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", 569)
    c2.metric("Input Features", 30)
    c3.metric("Models Compared", 5)

    st.subheader("Models Implemented")
    st.write(
        "Logistic Regression, Decision Tree, K-Nearest Neighbors, "
        "Gaussian Naive Bayes and Random Forest."
    )

    st.subheader("Evaluation Metrics")
    st.write("Accuracy, AUC, Precision, Recall, F1 Score and MCC Score.")

    st.subheader("Overall Comparison")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

elif page == "Dataset Explorer":
    st.subheader("Test Dataset Preview")
    st.write(
        "The application uses only the held-out test dataset for interactive evaluation, "
        "which keeps the deployed app lightweight."
    )
    st.dataframe(test_df.head(20), use_container_width=True)

    st.subheader("Class Distribution")
    counts = test_df["target"].value_counts().rename(index={0: "Malignant", 1: "Benign"})
    st.bar_chart(counts)

    st.subheader("Feature Summary")
    feature_cols = [c for c in test_df.columns if c != "target"]
    st.dataframe(test_df[feature_cols].describe().T, use_container_width=True)

elif page == "Model Evaluation":
    st.subheader("Evaluate a Saved Model")
    selected_model = st.selectbox("Select a model", list(MODEL_FILES.keys()))

    model_bundle = load_model(selected_model)
    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")
    feature_names = model_bundle["feature_names"]

    X_test = test_df[feature_names]
    y_test = test_df["target"]

    X_input = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_input)
    y_scores, score_kind, score_error = get_positive_class_scores(model, X_input)
    auc = roc_auc_score(y_test, y_scores) if y_scores is not None else np.nan

    if score_error:
        st.warning(
            "Model probability output is unavailable for this artifact. "
            f"Details: {score_error}"
        )
    elif score_kind == "score":
        st.info("AUC is computed from decision scores because probabilities are unavailable.")

    values = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    cols = st.columns(6)
    for col, (name, value) in zip(cols, values.items()):
        col.metric(name, f"{value:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Malignant", "Actual Benign"],
        columns=["Predicted Malignant", "Predicted Benign"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=["Malignant", "Benign"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

elif page == "Test Data Prediction":
    st.subheader("Upload Test CSV")
    st.write(
        "Upload a CSV containing the same 30 input features. "
        "If no file is uploaded, the bundled test_data.csv is used."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    prediction_df = pd.read_csv(uploaded) if uploaded is not None else test_df.copy()

    selected_model = st.selectbox("Choose model for prediction", list(MODEL_FILES.keys()))
    bundle = load_model(selected_model)
    model = bundle["model"]
    scaler = bundle.get("scaler")
    feature_names = bundle["feature_names"]

    missing = [c for c in feature_names if c not in prediction_df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        X = prediction_df[feature_names]
        X_input = scaler.transform(X) if scaler is not None else X
        pred = model.predict(X_input)
        y_scores, score_kind, score_error = get_positive_class_scores(model, X_input)

        result = prediction_df.copy()
        result["predicted_class"] = np.where(pred == 1, "Benign", "Malignant")

        if score_kind == "probability":
            result["benign_probability"] = y_scores
        elif score_kind == "score":
            result["benign_score"] = y_scores

        if score_error:
            st.warning(
                "Model probability output is unavailable for this artifact. "
                f"Details: {score_error}"
            )
        elif score_kind == "score":
            st.info("Prediction output includes decision scores because probabilities are unavailable.")

        st.success(f"Predictions completed using {selected_model}.")
        st.dataframe(result.head(50), use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
