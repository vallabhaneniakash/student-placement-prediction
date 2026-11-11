"""
Student Placement Prediction Using Machine Learning
---------------------------------------------------
Trains a Random Forest classifier on dataset.csv, prints the accuracy
and confusion matrix, and saves the trained model for the Streamlit app.
"""

import joblib
import matplotlib

matplotlib.use("Agg")  # lets the script save the plot without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# 1. Load the dataset
df = pd.read_csv("dataset.csv")
print("Dataset shape:", df.shape)
print(df.head(), "\n")

# 2. Basic preprocessing
print("Missing values per column:")
print(df.isnull().sum(), "\n")
df = df.dropna()            # remove rows with missing values (none expected)
df = df.drop_duplicates()   # remove duplicate rows

# 3. Split features (X) and target (y)
X = df.drop("Placed", axis=1)
y = df["Placed"]

# 4. Train / test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train the Random Forest model
# (Random Forests do not need feature scaling, so we skip it.)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print(cm, "\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

# 7. Save the confusion matrix as an image
ConfusionMatrixDisplay(cm, display_labels=["Not Placed", "Placed"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

# 8. Save the model and its accuracy for the Streamlit app
joblib.dump({"model": model, "accuracy": accuracy, "cm": cm}, "model.pkl")
print("Model saved as model.pkl")
