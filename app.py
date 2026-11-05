"""Interactive evaluation app for the five saved Dry Bean classifiers."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_DIR = PROJECT_DIR / "model"


st.set_page_config(
    page_title="Dry Bean Classifier Evaluation",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 6px;
        padding: 0.75rem 0.9rem;
    }
    .small-note {color: #666; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metadata() -> dict:
    with (ARTIFACT_DIR / "metadata.json").open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_bundled_test_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_DIR / "test_data.csv")


@st.cache_resource
def load_models(model_files: dict[str, str]) -> dict[str, object]:
    return {
        model_name: joblib.load(MODEL_DIR / filename)
        for model_name, filename in model_files.items()
    }


def validate_test_data(
    data: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = feature_columns + [target_column]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "The uploaded CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if data[required_columns].isna().any().any():
        raise ValueError("The uploaded CSV contains missing values.")

    X = data[feature_columns].copy()
    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="raise")
    y = data[target_column].astype(str).str.strip()
    return X, y


def evaluate_model(model: object, X: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    class_order = model.classes_

    try:
        auc = roc_auc_score(
            y,
            probabilities,
            labels=class_order,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision_score(
            y, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, average="weighted", zero_division=0
        ),
        "F1": f1_score(
            y, predictions, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return {
        "metrics": metrics,
        "predictions": predictions,
        "classes": class_order,
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=class_order
        ),
        "classification_report": classification_report(
            y,
            predictions,
            labels=class_order,
            output_dict=True,
            zero_division=0,
        ),
    }


metadata = load_metadata()
feature_columns = metadata["feature_columns"]
target_column = metadata["target_column"]
models = load_models(metadata["model_files"])

st.title("Dry Bean Classification")
st.write(
    "Evaluate five pre-trained classification models on labelled test data. "
    "The models are loaded from saved artifacts and are never retrained in this app."
)

with st.sidebar:
    st.header("Evaluation controls")
    uploaded_file = st.file_uploader(
        "Upload labelled test data",
        type=["csv"],
        help=(
            "The CSV must contain all 16 feature columns and the Class target "
            "column. Use the repository test_data.csv as the reference format."
        ),
    )
    selected_model = st.selectbox(
        "Detailed model view",
        options=list(models.keys()),
    )
    st.markdown(
        f"""
        <div class="small-note">
        Required rows: one or more<br>
        Required features: {len(feature_columns)}<br>
        Target column: {target_column}
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded_file is None:
    test_data = load_bundled_test_data()
    st.info(
        "Using the bundled test_data.csv. Upload another labelled CSV from the "
        "sidebar to evaluate the same saved models on different test records."
    )
else:
    test_data = pd.read_csv(uploaded_file)
    st.success(f"Loaded uploaded file: {uploaded_file.name}")

try:
    X_test, y_test = validate_test_data(
        test_data, feature_columns, target_column
    )
except (ValueError, TypeError) as error:
    st.error(str(error))
    st.stop()

unknown_classes = sorted(set(y_test) - set(metadata["dataset"]["classes"]))
if unknown_classes:
    st.error(
        "The target column contains unknown classes: "
        + ", ".join(unknown_classes)
    )
    st.stop()

st.caption(
    f"Evaluation rows: {len(test_data):,} | Input features: "
    f"{len(feature_columns)} | Target classes present: {y_test.nunique()}"
)

evaluations = {
    model_name: evaluate_model(model, X_test, y_test)
    for model_name, model in models.items()
}

st.header("All-model comparison")
comparison_rows = []
for model_name, evaluation in evaluations.items():
    comparison_rows.append(
        {
            "Model": model_name,
            **{
                metric_name: metric_value
                for metric_name, metric_value in evaluation["metrics"].items()
            },
        }
    )
comparison = pd.DataFrame(comparison_rows).sort_values(
    by=["F1", "MCC", "Accuracy"], ascending=False
)
st.dataframe(
    comparison.style.format(
        {
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

st.header(f"Detailed evaluation: {selected_model}")
selected_evaluation = evaluations[selected_model]
metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metric_columns = st.columns(3)
for index, metric_name in enumerate(metric_names):
    value = selected_evaluation["metrics"][metric_name]
    formatted_value = "Not available" if pd.isna(value) else f"{value:.4f}"
    metric_columns[index % 3].metric(metric_name, formatted_value)

left_column, right_column = st.columns([1.05, 1.35])
with left_column:
    st.subheader("Confusion matrix")
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        selected_evaluation["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=selected_evaluation["classes"],
        yticklabels=selected_evaluation["classes"],
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with right_column:
    st.subheader("Classification report")
    report = pd.DataFrame(
        selected_evaluation["classification_report"]
    ).transpose()
    display_rows = [
        row
        for row in list(selected_evaluation["classes"])
        + ["accuracy", "macro avg", "weighted avg"]
        if row in report.index
    ]
    st.dataframe(
        report.loc[display_rows].style.format(
            {
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}",
            }
        ),
        width="stretch",
    )

prediction_output = test_data.copy()
prediction_output["Predicted_Class"] = selected_evaluation["predictions"]
st.download_button(
    "Download selected-model predictions",
    data=prediction_output.to_csv(index=False).encode("utf-8"),
    file_name="dry_bean_predictions.csv",
    mime="text/csv",
)

with st.expander("Expected CSV columns"):
    st.code(", ".join(feature_columns + [target_column]), language=None)
