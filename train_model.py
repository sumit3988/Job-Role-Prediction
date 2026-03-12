"""
train_model.py
==============
Trains and compares four ML models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Support Vector Machine (SVM)

Evaluates each on accuracy, precision, recall, F1-score.
Selects the best model by accuracy and saves it as model.pkl.
Also generates visualization charts.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from data_preprocessing import preprocess

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------
# 1. Load preprocessed data
# ---------------------------------------------------------------
print("=" * 60)
print("  JOB ROLE PREDICTION – MODEL TRAINING")
print("=" * 60)

X_train, X_test, y_train, y_test, cat_encoders, target_encoder = preprocess()
feature_names = list(X_train.columns)
class_names = list(target_encoder.classes_)

# ---------------------------------------------------------------
# 2. Define models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=15, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=20,
                                                   random_state=42, n_jobs=-1),
    "SVM":                 SVC(kernel="rbf", C=10, gamma="scale",
                               random_state=42, probability=True),
}

# ---------------------------------------------------------------
# 3. Train & Evaluate
# ---------------------------------------------------------------
results = []

for name, model in models.items():
    print(f"\n[TRAINING] {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": name, "Accuracy": acc,
        "Precision": prec, "Recall": rec, "F1-Score": f1,
        "_model_obj": model, "_y_pred": y_pred,
    })

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# ---------------------------------------------------------------
# 4. Select best model
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df["Accuracy"].idxmax()]
best_model_name = best_row["Model"]
best_model      = best_row["_model_obj"]

print("=" * 60)
print(f"  BEST MODEL: {best_model_name}  (Accuracy={best_row['Accuracy']:.4f})")
print("=" * 60)

# ---------------------------------------------------------------
# 5. Save best model & feature names
# ---------------------------------------------------------------
joblib.dump(best_model, "model.pkl")
joblib.dump(feature_names, "artifacts/feature_names.pkl")
print("[INFO] model.pkl saved.")

# ---------------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------------

# ── 6a. Job Role Distribution ──────────────────────────────────
def plot_job_role_distribution():
    df_raw = pd.read_csv("dataset.csv")
    role_counts = df_raw["Job_Role"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("husl", len(role_counts))
    bars = ax.bar(role_counts.index, role_counts.values, color=palette, edgecolor="white", linewidth=0.8)

    # Annotate bars
    for bar, val in zip(bars, role_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Job Role Distribution in Dataset", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Job Role", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("plots/job_role_distribution.png", dpi=150)
    plt.close()
    print("[INFO] Saved plots/job_role_distribution.png")


# ── 6b. Feature Importance (Random Forest) ────────────────────
def plot_feature_importance():
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # top 20

    rev_indices = list(reversed(indices))
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = sns.color_palette("viridis", 20)
    ax.barh(
        [feature_names[i] for i in rev_indices],
        importances[rev_indices],
        color=palette, edgecolor="white"
    )
    ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()
    print("[INFO] Saved plots/feature_importance.png")


# ── 6c. Model Comparison ──────────────────────────────────────
def plot_model_comparison():
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = results_df["Model"].tolist()
    x = np.arange(len(model_names))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, metric in enumerate(metrics):
        vals = results_df[metric].tolist()
        bars = ax.bar(x + i * width, vals, width, label=metric,
                      color=colors[i], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Model Comparison – Metrics", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150)
    plt.close()
    print("[INFO] Saved plots/model_comparison.png")


# ── 6d. Confusion Matrix for best model ───────────────────────
def plot_confusion_matrix():
    best_y_pred = best_row["_y_pred"]
    cm = confusion_matrix(y_test, best_y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title(f"Confusion Matrix – {best_model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.close()
    print("[INFO] Saved plots/confusion_matrix.png")


plot_job_role_distribution()
plot_feature_importance()
plot_model_comparison()
plot_confusion_matrix()

# ---------------------------------------------------------------
# 7. Summary table
# ---------------------------------------------------------------
display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].copy()
display_df = display_df.round(4)
print("\n[SUMMARY]\n")
print(display_df.to_string(index=False))
print(f"\nBest Model → {best_model_name} saved as model.pkl")
print("All plots saved to plots/ directory.")
