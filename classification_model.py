import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# STEP 1: Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2: Load the dataset
df = pd.read_csv(DATA_PATH)

print("=== DATASET PREVIEW ===")
print(df.head())
print("\nDataset shape:", df.shape)

# STEP 3: Prepare data
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")

df = df.dropna(subset=["Marks", "Department", "Age"]).copy()

# STEP 4: Create performance categories
# These thresholds are for demonstration only.
df["Performance"] = df["Marks"].apply(
    lambda mark: "High" if mark >= 85 else "Low"
)

print("\n=== PERFORMANCE CATEGORIES ===")
print(df["Performance"].value_counts())

# STEP 5: Select features and target
# Marks is excluded to avoid target leakage.
X = df[["Age", "Department"]]
y = df["Performance"]

# STEP 6: Validate the dataset
class_counts = y.value_counts()

if y.nunique() < 2:
    raise ValueError(
        "Classification requires at least two different categories."
    )

smallest_class = int(class_counts.min())

if smallest_class < 2:
    raise ValueError(
        "Each category needs at least two records. "
        "Please add more representative data."
    )

# Choose a fold count that each class can support.
n_splits = min(3, smallest_class)

print(f"\nUsing {n_splits}-fold stratified cross-validation.")

# STEP 7: Prepare numerical and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, ["Age"]),
    ("categorical", categorical_pipeline, ["Department"])
])

# STEP 8: Build the classification model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# STEP 9: Generate cross-validated predictions
cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

y_pred = cross_val_predict(model, X, y, cv=cv)

print("\n=== ACTUAL VS PREDICTED ===")
for actual, predicted in zip(y, y_pred):
    print(f"Actual: {actual} | Predicted: {predicted}")

# STEP 10: Evaluate the predictions
accuracy = accuracy_score(y, y_pred)

report = classification_report(
    y,
    y_pred,
    labels=["High", "Low"],
    zero_division=0
)

print("\n=== CLASSIFICATION REPORT ===")
print(report)
print(f"Cross-validated accuracy: {accuracy:.2%}")

# STEP 11: Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y,
    y_pred,
    labels=["High", "Low"],
    display_labels=["High", "Low"],
    cmap="Blues"
)

plt.title("Student Performance Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

# STEP 12: Save report
report_path = OUTPUT_DIR / "classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("CLASSIFICATION MODEL RESULTS\n")
    file.write("=" * 35 + "\n")
    file.write("Algorithm: Logistic Regression\n")
    file.write("Features: Age, Department\n")
    file.write("Target: Performance\n")
    file.write("Category rule: Marks >= 85 is High; otherwise Low\n")
    file.write(f"Dataset rows used: {len(df)}\n")
    file.write(f"Cross-validation folds: {n_splits}\n")
    file.write(f"Cross-validated accuracy: {accuracy:.2%}\n\n")
    file.write(report)

print("\nReport saved to:", report_path)
print("Confusion matrix saved successfully.")