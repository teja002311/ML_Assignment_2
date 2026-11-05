"""Train and persist the five classifiers required for ML Assignment 2."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "Class"
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset"
UCI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
)
ARFF_MEMBER = "DryBeanDataset/Dry_Bean_Dataset.arff"

FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def load_dry_bean_dataset() -> pd.DataFrame:
    """Load the official UCI ARFF file directly into memory."""
    with urlopen(UCI_DOWNLOAD_URL, timeout=60) as response:
        archive_bytes = BytesIO(response.read())

    with ZipFile(archive_bytes) as archive:
        with archive.open(ARFF_MEMBER) as arff_file:
            data = pd.read_csv(
                arff_file,
                skiprows=25,
                names=FEATURE_COLUMNS + [TARGET_COLUMN],
            )

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.strip()
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    return data


def build_models() -> dict[str, BaseEstimator]:
    """Create the five required classifiers with reproducible settings."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                    ),
                ),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=3.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(
    model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calculate every metric required by the assignment."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    class_order = model.classes_

    scores = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(
            y_test,
            probabilities,
            labels=class_order,
            multi_class="ovr",
            average="weighted",
        ),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    matrix = confusion_matrix(y_test, predictions, labels=class_order)
    return scores, predictions, matrix


def observation_for(model_name: str, metrics: dict[str, float]) -> str:
    """Create a concise model-specific observation for the report."""
    f1 = metrics["F1"]
    if model_name == "Logistic Regression":
        return (
            f"Strong linear baseline (weighted F1 {f1:.4f}); scaling and balanced "
            "class weights support stable multiclass performance."
        )
    if model_name == "Decision Tree":
        return (
            f"Interpretable nonlinear rules (weighted F1 {f1:.4f}), but performance "
            "is more sensitive to depth and leaf-size choices."
        )
    if model_name == "K-Nearest Neighbors":
        return (
            f"Captures local class neighborhoods well after scaling (weighted F1 "
            f"{f1:.4f}); prediction cost grows with the stored training set."
        )
    if model_name == "Gaussian Naive Bayes":
        return (
            f"Fast probabilistic baseline (weighted F1 {f1:.4f}); its conditional "
            "independence and Gaussian assumptions limit performance."
        )
    return (
        f"RBF decision boundaries provide strong nonlinear separation (weighted F1 "
        f"{f1:.4f}) at a higher training and inference cost."
    )


def main(project_dir: Path | None = None) -> pd.DataFrame:
    """Train models, save all artifacts, and return the comparison table."""
    project_dir = project_dir or Path(__file__).resolve().parent
    model_dir = project_dir / "model"
    artifact_dir = project_dir / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data = load_dry_bean_dataset()
    raw_instance_count = len(data)
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Raw dataset shape: ({raw_instance_count}, {data.shape[1]})")
    print(f"Modeling dataset shape after duplicate removal: {data.shape}")
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print("Class distribution:")
    print(data[TARGET_COLUMN].value_counts().sort_index())

    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.to_numpy()
    test_data.to_csv(project_dir / "test_data.csv", index=False)

    model_results: dict[str, dict[str, object]] = {}
    confusion_matrices: dict[str, list[list[int]]] = {}
    rows: list[dict[str, object]] = []

    for model_name, model in build_models().items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        metrics, _, matrix = calculate_metrics(model, X_test, y_test)
        joblib.dump(model, model_dir / MODEL_FILES[model_name])

        rounded_metrics = {
            metric_name: round(float(value), 6)
            for metric_name, value in metrics.items()
        }
        observation = observation_for(model_name, metrics)
        model_results[model_name] = {
            "model_file": MODEL_FILES[model_name],
            "metrics": rounded_metrics,
            "observation": observation,
        }
        confusion_matrices[model_name] = matrix.tolist()
        rows.append(
            {
                "ML Model Name": model_name,
                **rounded_metrics,
                "Observation": observation,
            }
        )
        print(
            ", ".join(
                f"{metric_name}={metric_value:.4f}"
                for metric_name, metric_value in metrics.items()
            )
        )

    comparison = pd.DataFrame(rows).sort_values(
        by=["F1", "MCC", "Accuracy"], ascending=False
    )
    comparison.to_csv(artifact_dir / "model_comparison.csv", index=False)

    winner = str(comparison.iloc[0]["ML Model Name"])
    metadata = {
        "dataset": {
            "name": "Dry Bean Dataset",
            "source": UCI_DATASET_PAGE,
            "instances": int(len(data)),
            "raw_instances": int(raw_instance_count),
            "duplicates_removed": duplicate_count,
            "features": len(FEATURE_COLUMNS),
            "target": TARGET_COLUMN,
            "classes": sorted(y.unique().tolist()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "random_state": RANDOM_STATE,
        },
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "model_files": MODEL_FILES,
        "models": model_results,
        "winner": winner,
        "confusion_matrices": confusion_matrices,
        "metric_average": "weighted",
        "auc_strategy": "one-vs-rest, weighted",
    }
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nModel comparison:")
    print(
        comparison[
            ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        ].to_string(index=False)
    )
    print(f"\nOverall winner by weighted F1: {winner}")
    print(f"Saved test data to: {project_dir / 'test_data.csv'}")
    print(f"Saved trained models to: {model_dir}")
    return comparison


if __name__ == "__main__":
    main()
