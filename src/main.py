import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folders automatically (FIX ERROR)
os.makedirs("outputs/plots", exist_ok=True)
from sklearn.model_selection import cross_val_score
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# Load Dataset
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10,6))
sns.countplot(x=y)
plt.title("Target Distribution")
plt.savefig("outputs/plots/target_distribution.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title("Feature Correlation")
plt.savefig("outputs/plots/correlation_heatmap.png")
plt.close()

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
   model.fit(X_train, y_train)
   pred = model.predict(X_test)
    # -----------------------------
# Feature Importance (Random Forest)
# -----------------------------
rf_model = models["Random Forest"]

importances = rf_model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.savefig("outputs/plots/feature_importance.png")
plt.close()
# -----------------------------
# Cross Validation
# -----------------------------
scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("Cross-validation Accuracy:", scores.mean())
# -----------------------------
# Training & Evaluation
# -----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# -----------------------------
# Best Model Selection
# -----------------------------
best_model_name = max(results, key=results.get)
print("\nBest Model:", best_model_name)

best_model = models[best_model_name]
# -----------------------------
# Model Comparison Plot
# -----------------------------
plt.figure()
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.title("Model Comparison")
plt.savefig("outputs/plots/model_comparison.png")
plt.close()

# -----------------------------
# Confusion Matrix
# -----------------------------
pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("outputs/plots/confusion_matrix.png")
plt.close()

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# -----------------------------
# ROC Curve
# -----------------------------
y_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.legend()
plt.title("ROC Curve")
plt.savefig("outputs/plots/roc_curve.png")
plt.close()

# -----------------------------
# -----------------------------
# Hyperparameter Tuning
# -----------------------------
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
# -----------------------------
# Save Model
# -----------------------------
joblib.dump(grid.best_estimator_, "outputs/best_model.pkl")
print("Model saved successfully!")