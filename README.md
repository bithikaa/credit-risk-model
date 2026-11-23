# credit-risk-model
# Credit Risk Modeling & Prediction

An end-to-end machine learning project using the **German Credit Dataset** to classify loan applicants as good or bad credit risks. The project covers data preprocessing, exploratory data analysis, feature engineering, handling class imbalance, model training, hyperparameter tuning, evaluation, and deployment using Streamlit.

## Project Overview

- **Dataset:** German Credit Dataset
- **Records:** 1,000 applicants
- **Target distribution:** 700 Good Risk (70%) / 300 Bad Risk (30%)
- **Models:** Decision Tree, Random Forest, Extra Trees, XGBoost, Logistic Regression
- **Final selected model:** Random Forest
- **Deployment:** Streamlit web application

## Tech Stack

- **Languages:** Python, SQL
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Imbalanced Data:** SMOTE (Imbalanced-learn)
- **Hyperparameter Tuning:** GridSearchCV
- **Model Persistence:** Joblib
- **Deployment:** Streamlit

## Dataset & Exploratory Data Analysis

The German Credit Dataset contains 1,000 loan applicants and demographic and financial attributes.

**Features used for modeling:**
- Age
- Sex
- Job
- Housing
- Saving accounts
- Checking account
- Credit amount
- Duration

**Key EDA findings:**

- Applicants classified as bad risk had higher average credit amounts (~3,938) than good-risk applicants (~2,985).
- Bad-risk applicants had longer average loan durations (~25 months vs. ~19 months).
- Credit amount and loan duration showed a positive correlation of approximately 0.62.
- Missing values in Saving accounts (18.3%) and Checking account (39.4%) were handled using an `unknown` category.

## Data Preprocessing

The following preprocessing steps were implemented:

1. Removed the unnecessary index column.
2. Handled missing categorical values using the `unknown` category.
3. Selected eight relevant features for model training.
4. Applied Label Encoding to categorical features and the target variable.
5. Performed an 80:20 stratified train-test split using `random_state=42`.
6. Applied SMOTE to the training data to address class imbalance.

**Target encoding:** Good Risk = 1, Bad Risk = 0.

## Model Development & Hyperparameter Tuning

Four classification algorithms were implemented and evaluated:

| Model | Hyperparameter tuning |
|---|---|
| Decision Tree | GridSearchCV |
| Random Forest | GridSearchCV |
| Extra Trees | GridSearchCV |
| XGBoost | GridSearchCV |

Hyperparameter tuning was performed using 5-fold cross-validation with accuracy as the scoring metric.

**Best hyperparameters:**

| Model | Selected parameters |
|---|---|
| Decision Tree | max_depth=10, min_samples_leaf=4 |
| Random Forest | n_estimators=100, max_depth=10, min_samples_leaf=2 |
| Extra Trees | n_estimators=200, max_depth=10, min_samples_split=5 |
| XGBoost | n_estimators=200, max_depth=3, learning_rate=0.2 |

## Model Evaluation & Results

Models were evaluated on a held-out test set of 200 records using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Random Forest | **73.5%** | **0.7714** |
| XGBoost | 73.5% | 0.7525 |
| Extra Trees | 71.5% | — |
| Logistic Regression | 68.5% | 0.736 |
| Decision Tree | 66.5% | — |


### Confusion Matrices

**Random Forest**
```text
[[43, 17],
 [36, 104]]
```

**XGBoost**
```text
[[35, 25],
 [28, 112]]
```

Rows represent actual classes (Bad, Good), while columns represent predicted classes (Bad, Good).

### Final Model Selection

**Random Forest** was selected based on its higher ROC-AUC score and its overall balance between identifying bad-risk and good-risk applicants.

It achieved:
- Accuracy: 73.5%
- ROC-AUC: 0.7714
- Bad-risk recall: 71.7%
- Good-risk recall: 74.3%

The trained model was saved as `random_forest_credit_model.pkl` using Joblib.

## Streamlit Application

A Streamlit web application was developed to demonstrate the trained model through an interactive prediction interface.

**Application features:**
- User inputs for all eight model features.
- Categorical feature encoding using saved LabelEncoders.
- Credit risk classification (Good Risk / Bad Risk).
- Model prediction confidence, where supported by `predict_proba`.

### Run Locally

Clone the repository:

```bash
git clone https://github.com/bithikaa/credit-risk-model.git
cd credit-risk-model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

Ensure the trained model and encoder `.pkl` files are present in the project directory.

## Limitations & Future Improvements

- Explore One-Hot Encoding and preprocessing pipelines for improved categorical feature handling.
- Evaluate cost-sensitive learning to account for the different costs of credit risk misclassification.
- Implement SHAP or LIME for model interpretability.
- Apply probability calibration for more reliable probability estimates.
- Explore larger datasets and additional financial attributes.
- Develop a REST API using FastAPI for model integration.

---

**Disclaimer:** This project is intended for educational purposes and demonstrates machine learning techniques for credit risk classification. It is not designed for actual lending decisions.
