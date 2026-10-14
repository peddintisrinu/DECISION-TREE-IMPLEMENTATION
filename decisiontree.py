# ============================================
# 🌳 Decision Tree Implementation (Full Code)
# ============================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================
# Step 2: Load Dataset
# ============================================
data = load_iris()

X = data.data
y = data.target

print("Dataset Loaded Successfully!\n")
print("Feature Names:", data.feature_names)
print("Target Names:", data.target_names)
print("Shape of Features:", X.shape)
print("Shape of Target:", y.shape)

# ============================================
# Step 3: Split Dataset
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nData Split Completed!")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ============================================
# Step 4: Train Decision Tree Model
# ============================================
model = DecisionTreeClassifier(
    criterion='gini',   # you can also use 'entropy'
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ============================================
# Step 5: Predictions
# ============================================
y_pred = model.predict(X_test)

# ============================================
# Step 6: Model Evaluation
# ============================================
print("\n========== MODEL EVALUATION ==========\n")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================
# Step 7: Visualize Decision Tree
# ============================================
plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True
)

plt.title("Decision Tree Visualization")
plt.show()

# ============================================
# Step 8: Feature Importance (Extra)
# ============================================
importance = model.feature_importances_

print("\nFeature Importances:")
for i, v in enumerate(importance):
    print(f"{data.feature_names[i]}: {v:.4f}")

# ============================================
# Step 9: Predict New Sample (Optional)
# ============================================
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
prediction = model.predict(sample)

print("\nPrediction for sample:", sample)
print("Predicted Class:", data.target_names[prediction][0])
