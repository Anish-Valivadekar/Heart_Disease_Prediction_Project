
# Project: Heart Disease Risk Prediction

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)

import warnings
warnings.filterwarnings("ignore")


# 2. Load Dataset

data = pd.read_csv("/content/Heart Disease data.csv")
data.head()

print("Dataset Shape:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())
print("\nData Info:\n")
print(data.info())


# 3. Exploratory Data Analysis


# Statistical Summary
print("\nStatistical Summary:\n", data.describe())

# Class Distribution
sns.countplot(x='target', data=data)
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 4. Data Preprocessing


X = data.drop("target", axis=1)
y = data["target"]

# Stratified Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Model Building


models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


# 6. Cross Validation 


rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)

print("\nRandom Forest Cross-Validation Accuracy:", cv_scores.mean())


# 7. Hyperparameter Tuning (GridSearch)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid.fit(X_train_scaled, y_train)

best_rf = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)


# 8. Final Model Evaluation

y_pred_rf = best_rf.predict(X_test_scaled)
y_prob_rf = best_rf.predict_proba(X_test_scaled)[:,1]

print("\nFinal Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob_rf)
print("\nROC-AUC Score:", roc_auc)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 9. ROC Curve


fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# 10. Feature Importance


feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features:\n", feature_importance)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()


# 11. Final Model Comparison

print("\nModel Comparison:")
for model, score in results.items():
    print(f"{model}: {score:.4f}")

print("\nBest Model Selected: Random Forest (Based on Accuracy & ROC-AUC)")
