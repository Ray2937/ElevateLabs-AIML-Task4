import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Load and preprocess the dataset
print('--- 1. LOADING & PREPROCESSING DATA ---')
df = pd.read_csv('data.csv')

# Drop unnecessary columns like 'id' and the empty 'Unnamed: 32' column if present
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Convert diagnosis target ('M' -> 1, 'B' -> 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Select features (X) and target (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 2. Train/test split and feature standardization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Fit a Logistic Regression model
print('\n--- 3. FITTING LOGISTIC REGRESSION MODEL ---')
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions and probabilities
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 4. Evaluate with confusion matrix, classification report, and ROC-AUC
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print('Confusion Matrix:\n', cm)
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc:.4f}')

# Save Confusion Matrix plot
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=ax, cmap='Blues', colorbar=False
)
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('confusion_matrix.png')
plt.close()

# Save ROC Curve plot
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label=f'ROC curve (AUC = {roc_auc:.2f})',
)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

print(
    '\nTask 4 code executed successfully! Visualizations saved as'
    ' confusion_matrix.png and roc_curve.png'
)