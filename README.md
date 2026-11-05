# Dry Bean Classification - ML Assignment 2

**Student:** P Tejeswara Varma  
**BITS ID:** 2026AJ05571

## a. Problem statement

The objective is to classify dry bean seeds into one of seven varieties using
16 numerical shape and size measurements extracted from bean images. Five
classification algorithms are trained on the same stratified training split,
evaluated on the same untouched test split, and exposed through a Streamlit
application that loads saved models without retraining.

## b. Dataset description

- **Dataset:** Dry Bean Dataset
- **Public source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Raw instances:** 13,611
- **Exact duplicates removed:** 68
- **Instances used for modelling:** 13,543
- **Input features:** 16 numerical features
- **Target:** `Class`
- **Classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA
- **Split:** Stratified 80% training and 20% test, `random_state=42`
- **Test file:** `test_data.csv` contains all required input features and the target column

The features describe bean geometry, including area, perimeter, major and minor
axis lengths, eccentricity, convex area, diameter, extent, solidity, roundness,
compactness and four shape factors. The raw dataset contains no missing values.

## c. Project links

- **GitHub repository:** [ML Assignment 2](https://github.com/teja002311/ML_Assignment_2)
- **Live Streamlit app:** Pending deployment

## d. Models used and comparison

All reported multiclass precision, recall and F1 values use weighted averaging.
AUC uses weighted one-vs-rest calculation.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.914729 | 0.993185 | 0.916246 | 0.914729 | 0.914980 | 0.897309 |
| Decision Tree | 0.892211 | 0.956660 | 0.893148 | 0.892211 | 0.892501 | 0.869859 |
| K-Nearest Neighbors | 0.913621 | 0.983471 | 0.914400 | 0.913621 | 0.913803 | 0.895555 |
| Gaussian Naive Bayes | 0.897010 | 0.989880 | 0.899653 | 0.897010 | 0.897199 | 0.876171 |
| Support Vector Machine | **0.926172** | **0.993310** | **0.926886** | **0.926172** | **0.926310** | **0.910912** |

## Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Produced a strong linear baseline. Scaling and balanced class weights gave stable results, but linear boundaries could not capture all class overlap. |
| Decision Tree | Learned interpretable nonlinear rules but had the lowest AUC and F1, showing greater sensitivity to the chosen tree depth and leaf size. |
| K-Nearest Neighbors | Performed close to logistic regression after scaling. It captured local neighbourhoods effectively, although inference requires retaining the training samples. |
| Gaussian Naive Bayes | Trained quickly and achieved high AUC, but its Gaussian conditional-independence assumptions reduced classification accuracy and F1. |
| Support Vector Machine | Achieved the best accuracy, weighted F1 and MCC. The RBF kernel modelled nonlinear class boundaries more effectively than the other classifiers. |
| **Overall winner** | **Support Vector Machine**, selected by weighted F1 (0.926310), with the highest accuracy and MCC as supporting evidence. |

## Repository structure

```text
ML_Assignment_2/
├── app.py
├── train_models.py
├── 2026AJ05571_ML_Assignment_2.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── support_vector_machine.joblib
└── artifacts/
    ├── metadata.json
    └── model_comparison.csv
```

## Streamlit application features

- CSV test-data upload
- Bundled `test_data.csv` for immediate demonstration
- Dropdown for detailed model selection
- Six required evaluation metrics for all five models
- Confusion matrix for the selected model
- Per-class classification report
- Downloadable prediction file
- Saved-model loading only; the app never retrains on uploaded data

The uploaded CSV must include all 16 feature columns and the `Class` target
column. Unknown classes, missing columns and missing values are rejected with a
clear error message.

## Local execution

```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Model training is only needed when regenerating artifacts. The committed model
files and `test_data.csv` allow the Streamlit app to run directly after
dependency installation.

## Reproducibility

- Fixed random state: 42
- Duplicate records removed before splitting
- Stratified train/test split
- Preprocessing fitted only on training data through scikit-learn pipelines
- All five models evaluated on the identical test records

## BITS Virtual Lab evidence

The final submission PDF must include one screenshot showing successful
assignment execution on BITS Virtual Lab. This screenshot is added during the
final submission process.
