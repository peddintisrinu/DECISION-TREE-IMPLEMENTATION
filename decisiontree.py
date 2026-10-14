
# ============================================================
# DECISION TREE CLASSIFICATION - IRIS DATASET
# ============================================================

# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Step 2: Load Dataset
data = load_iris()

X = data.data
y = data.target

print("=" * 55)
print("       DECISION TREE CLASSIFICATION")
print("=" * 55)

print("\nDataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Feature names:", data.feature_names)
print("Target classes:", list(data.target_names))

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Create and Train Decision Tree Model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("                 MODEL EVALUATION")
print("=" * 55)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
        zero_division=0
    )
)

# Step 7: Visualize Confusion Matrix
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
).plot(cmap="Blues", values_format="d")

plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.show()

# Step 8: Visualize Decision Tree
plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    proportion=False,
    precision=2
)

plt.title("Decision Tree Classification - Iris Dataset")
plt.tight_layout()
plt.show()

# Step 9: Display Feature Importance
print("\n" + "=" * 55)
print("               FEATURE IMPORTANCE")
print("=" * 55)

for feature, importance in zip(
    data.feature_names,
    model.feature_importances_
):
    print(f"{feature}: {importance:.4f}")

# Feature importance visualization
plt.figure(figsize=(9, 5))

plt.barh(
    data.feature_names,
    model.feature_importances_
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Decision Tree")
plt.tight_layout()
plt.show()

# Step 10: Predict a New Sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(sample)
predicted_class = data.target_names[prediction[0]]

print("\n" + "=" * 55)
print("                NEW SAMPLE PREDICTION")
print("=" * 55)

print("\nInput Features:")
for name, value in zip(data.feature_names, sample[0]):
    print(f"{name}: {value} cm")

print("\nPredicted Flower Species:", predicted_class)

# Display prediction probabilities
probabilities = model.predict_proba(sample)[0]

print("\nPrediction Probabilities:")
for name, probability in zip(data.target_names, probabilities):
    print(f"{name}: {probability * 100:.2f}%")

print("\nProgram executed successfully!")

