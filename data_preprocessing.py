"""
data_preprocessing.py
=====================
Handles all data preprocessing steps for the Job Role Prediction project:
  - Loading data
  - Handling missing values
  - Feature engineering (multi-hot encoding for Skills)
  - Encoding categorical variables (Label Encoding / One-Hot)
  - Splitting into train/test sets (80/20)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
DATA_PATH = "dataset.csv"
TARGET_COL = "Job_Role"
RANDOM_STATE = 42

ALL_SKILLS = [
    "Python", "Java", "SQL", "AWS", "Machine Learning", "JavaScript",
    "React", "Node.js", "Docker", "Kubernetes", "TensorFlow", "Deep Learning",
    "C++", "PHP", "MongoDB", "Azure", "Linux", "Git"
]


# ---------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV dataset into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------
# 2. Handle Missing Values
# ---------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Numeric columns  → median
      - Categorical cols → 'Unknown'
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # Remove target from auto-fill list
    categorical_cols = [c for c in categorical_cols if c != TARGET_COL]

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[INFO] Filled missing values in '{col}' with median={median_val:.2f}")

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna("Unknown", inplace=True)
            print(f"[INFO] Filled missing values in '{col}' with 'Unknown'")

    return df


# ---------------------------------------------------------------
# 3. Feature Engineering – Multi-hot encode Skills column
# ---------------------------------------------------------------
def encode_skills(df: pd.DataFrame, skill_list: list = ALL_SKILLS) -> pd.DataFrame:
    """
    Convert comma-separated Skills string into binary columns (multi-hot).
    E.g., 'Python, SQL' → skill_Python=1, skill_SQL=1, skill_Java=0 …
    """
    for skill in skill_list:
        col_name = f"skill_{skill.replace(' ', '_').replace('.', '')}"
        df[col_name] = df["Skills"].apply(
            lambda x: 1 if isinstance(x, str) and skill in x else 0
        )
    df.drop(columns=["Skills"], inplace=True)
    print(f"[INFO] Skills encoded into {len(skill_list)} binary features")
    return df


# ---------------------------------------------------------------
# 4. Encode Categorical Variables
# ---------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame):
    """
    Label-encode Degree, Major, Certifications.
    Returns the modified DataFrame and a dict of fitted LabelEncoders.
    """
    cat_cols = ["Degree", "Major", "Certifications"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[INFO] Label-encoded '{col}': {list(le.classes_)}")

    return df, encoders


# ---------------------------------------------------------------
# 5. Encode Target Variable
# ---------------------------------------------------------------
def encode_target(df: pd.DataFrame):
    """Label-encode the Job_Role target column."""
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
    print(f"[INFO] Target classes: {list(le.classes_)}")
    return df, le


# ---------------------------------------------------------------
# 6. Train / Test Split
# ---------------------------------------------------------------
def split_data(df: pd.DataFrame):
    """Split features and target into 80/20 train-test sets."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------
def preprocess(path: str = DATA_PATH):
    """Full preprocessing pipeline. Returns X_train, X_test, y_train, y_test, encoders, target_encoder."""
    df = load_data(path)
    df = handle_missing_values(df)
    df = encode_skills(df)
    df, cat_encoders = encode_categoricals(df)
    df, target_encoder = encode_target(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Save encoders for use in app.py
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(cat_encoders, "artifacts/cat_encoders.pkl")
    joblib.dump(target_encoder, "artifacts/target_encoder.pkl")
    joblib.dump(list(df.drop(columns=[]).columns if TARGET_COL not in df.columns
                    else df.drop(columns=[TARGET_COL]).columns),
                "artifacts/feature_names.pkl")
    print("[INFO] Encoders saved to artifacts/")

    return X_train, X_test, y_train, y_test, cat_encoders, target_encoder


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, cat_enc, target_enc = preprocess()
    print("\nPreprocessing complete!")
    print(f"Feature columns ({X_train.shape[1]}): {list(X_train.columns)}")
