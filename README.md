# Task 4: Classification with Logistic Regression (Breast Cancer Prediction)

## Overview
This repository contains the solution for Task 4 of the AIML Internship at Elevate Labs. The objective of this task is to build a binary classification model using Logistic Regression, evaluate its performance using advanced classification metrics, and visualize the results.

## Files Included
- `task4.py`: The complete Python script for data preprocessing, scaling, model training, evaluation, and plotting.
- `data.csv`: The Breast Cancer Wisconsin dataset used for binary classification.
- `confusion_matrix.png`: Confusion matrix visualization displaying true vs. predicted classifications.
- `roc_curve.png`: Receiver Operating Characteristic (ROC) curve showing the model's diagnostic ability with the calculated AUC score.

## Workflow & Steps Performed
1. **Data Loading & Preprocessing:** Loaded the dataset (`data.csv`), dropped unnecessary identifier and empty columns, and mapped the target variable (`diagnosis`: 'M' -> 1, 'B' -> 0).
2. **Train-Test Split & Scaling:** Split the data into 80% training and 20% testing sets, and applied `StandardScaler` to standardize features for optimal gradient descent performance.
3. **Model Training:** Fitted a `LogisticRegression` classifier on the scaled training data.
4. **Model Evaluation:** Evaluated the model using key classification metrics:
   - **Confusion Matrix** (True Positives, True Negatives, False Positives, False Negatives)
   - **Precision, Recall, and F1-Score** via a classification report
   - **ROC-AUC Score** (Area Under the ROC Curve)
5. **Visualizations:** Generated and saved the Confusion Matrix plot and ROC-AUC curve plot to visually assess classification performance.
