# Intelligent Student Performance Prediction and Personalized Learning Recommendation System

## Project Overview

This project is an AI/ML-based Student Performance Prediction and Personalized Learning Recommendation System developed as part of the Trekverse internship assignment.

The system predicts student academic performance, identifies the level of support required, explains predictions using Explainable AI (SHAP), and provides personalized learning recommendations through a Streamlit web application.

## Dataset

The project uses the UCI Student Performance dataset.

Dataset file:

`student-mat.csv`

Dataset size:

- 395 students
- 33 columns

Target variable:

`G3` - Final Grade

The final grade ranges from 0 to 20.

G1 and G2 are intermediate grades and are excluded from the primary prediction model to avoid using later academic information as model input.

## Project Objectives

The main objectives are:

1. Analyze student performance data.
2. Perform data cleaning and validation.
3. Perform exploratory data analysis.
4. Perform feature engineering.
5. Predict final student grades.
6. Classify students according to support requirements.
7. Compare multiple machine learning algorithms.
8. Perform hyperparameter tuning.
9. Explain predictions using SHAP.
10. Identify student engagement profiles using clustering.
11. Generate personalized learning recommendations.
12. Perform robustness testing.
13. Deploy the system using Streamlit.
14. Provide downloadable prediction reports.
15. Perform automated unit testing.

## Support Categories

The project uses the following project-defined categories:

| Final Grade | Support Category |
|-------------|------------------|
| G3 < 10 | High Support |
| 10 <= G3 < 15 | Medium Support |
| G3 >= 15 | Low Support |

These categories are defined specifically for this project.

## Machine Learning Tasks

### Regression

The regression task predicts the student's final grade `G3`.

Algorithms used:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- K-Nearest Neighbors Regression

A tuned Random Forest Regressor is also used.

### Classification

The classification task predicts the student's support category.

Algorithms used:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors Classifier

A tuned Random Forest Classifier is also used.

### Clustering

K-Means clustering is used to identify student engagement profiles.

The final grade is not used as a clustering feature.

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to explain machine learning predictions.

The project provides:

- Global feature importance
- SHAP summary analysis
- Individual student explanation
- Top factors influencing predictions

## Personalized Recommendations

The recommendation module provides learning suggestions using information such as:

- Study time
- Previous failures
- Absences
- School support
- Family support
- Free time
- Social activities
- Lifestyle information
- Higher education goals
- Internet access

Examples include:

- Increase study time.
- Attend classes regularly.
- Practice previously failed subjects.
- Use additional academic support.
- Maintain a balance between social activities and study.
- Use reliable online educational resources.

## Project Structure

```text
Trekverse_Student_Performance/
│
├── data/
│   ├── student-mat.csv
│   ├── data_dictionary.csv
│   ├── cleaned_data.csv
│   ├── feature_engineered_data.csv
│   ├── X_train_processed.csv
│   ├── X_test_processed.csv
│   ├── y_reg_train.csv
│   ├── y_reg_test.csv
│   ├── y_cls_train.csv
│   ├── y_cls_test.csv
│   ├── regression_model_comparison.csv
│   └── classification_model_comparison.csv
│
├── models/
│   ├── preprocessor.pkl
│   ├── classification_preprocessor.pkl
│   ├── linear_regression.pkl
│   ├── decision_tree_regression.pkl
│   ├── random_forest_regression.pkl
│   ├── gradient_boosting_regression.pkl
│   ├── knn_regression.pkl
│   ├── logistic_regression_classification.pkl
│   ├── decision_tree_classification.pkl
│   ├── random_forest_classification.pkl
│   ├── gradient_boosting_classification.pkl
│   ├── knn_classification.pkl
│   ├── tuned_random_forest_regression.pkl
│   └── tuned_random_forest_classification.pkl
│
├── src/
│   ├── data_check.py
│   ├── data_analysis.py
│   ├── data_dictionary.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── ml_preprocessing.py
│   ├── regression.py
│   ├── multiple_regression.py
│   ├── regression_analysis.py
│   ├── classification.py
│   ├── classification_analysis.py
│   ├── clustering.py
│   ├── hyperparameter_tuning.py
│   ├── classification_tuning.py
│   ├── xai_analysis.py
│   ├── recommendation.py
│   ├── robustness_testing.py
│   │
│   └── app/
│       ├── app.py
│       └── pages/
│           ├── 2_Model_Performance.py
│           └── 3_Admin_Dashboard.py
│
├── tests/
│   └── test_project.py
│
├── eda_graphs/
│
├── requirements.txt
│
└── README.md