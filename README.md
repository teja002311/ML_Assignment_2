# Dry Bean Classification - ML Assignment 2

**Student:** P Tejeswara Varma  
**BITS ID:** 2026AJ05571

## a. Problem statement

For this assignment, I used measurements of dry bean seeds to predict their
variety. The dataset has seven bean classes and 16 input features. I trained the
five required classification models and compared them on the same test data.
The saved models are used in the Streamlit app.

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

The features are measurements such as area, perimeter, axis lengths,
eccentricity, roundness, compactness and shape factors. The dataset did not
contain any missing values. I removed 68 exact duplicate rows before splitting
the data.

## c. Project links

- **GitHub repository:** [ML Assignment 2](https://github.com/teja002311/ML_Assignment_2)
- **Live Streamlit app:** Pending deployment

## d. Models used and comparison

Precision, recall and F1 are weighted scores because this is a multiclass
dataset. AUC is calculated using weighted one-vs-rest.

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
| Logistic Regression | It gave good results, with an F1 score of 0.914980. Some class overlap was still difficult for its linear decision boundary. |
| Decision Tree | It was easy to interpret, but it gave the lowest AUC and F1 among the five models. |
| K-Nearest Neighbors | It performed close to Logistic Regression after feature scaling, with an F1 score of 0.913803. |
| Gaussian Naive Bayes | It trained quickly and had a high AUC, but its accuracy and F1 were lower than Logistic Regression, KNN and SVM. |
| Support Vector Machine | It gave the highest accuracy, F1 and MCC. The RBF kernel handled the overlapping classes better in this dataset. |
| **Overall winner** | **Support Vector Machine** is the winner because it achieved the highest weighted F1 score of 0.926310. |

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
- Included `test_data.csv` for testing
- Dropdown for detailed model selection
- Six required evaluation metrics for all five models
- Confusion matrix for the selected model
- Per-class classification report
- Downloadable prediction file
- Loads the saved models without retraining them

The uploaded CSV must include all 16 feature columns and the `Class` target
column. The app displays an error if any required column or value is missing.

## Local execution

```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The models are already saved in the `model` folder, so training is not required
to start the Streamlit app.
