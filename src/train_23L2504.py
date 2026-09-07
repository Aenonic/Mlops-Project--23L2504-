"""
train_23L2504.py
MLOps Assignment 1 - Version Control with Git, GitHub, and VS Code
Student ID: 23L2504

This script:
1. Loads a dataset from the data/ directory.
2. Trains a RandomForest model (regression - house price prediction).
3. Saves the trained model as a .pkl file into the model/ directory.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# --------------------------------------------------------------------
# Hyperparameters (Part 3 asks you to add one here, e.g. learning rate
# equivalent - for RandomForest we use n_estimators / max_depth)
# --------------------------------------------------------------------
N_ESTIMATORS = 200      # <-- this is the "hyperparameter variable" you will modify in Part 3
MAX_DEPTH = 10
RANDOM_STATE = 42

# Student identifier - required by assignment instructions
STUDENT_ID = "23L2504"

DATA_PATH = os.path.join("data", "dataset.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, f"model_{STUDENT_ID}.pkl")


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from the data/ directory."""
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded with shape: {df.shape}")
    return df


def train_model(df: pd.DataFrame):
    """Train a RandomForest model on the dataset."""
    # Assumes the last column is the target (e.g., house price).
    # Adjust the target_column name to match your actual dataset.
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Keep only numeric columns for this simple baseline
    X = X.select_dtypes(include=["number"])
    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)   # Feature Scaling

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("[INFO] Training RandomForestRegressor ...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"[INFO] Model trained. Test MAE: {mae:.4f}")

    return model


def save_model(model, path: str):
    """Serialize and save the trained model into the model/ directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")


def main():
    df = load_data(DATA_PATH)
    model = train_model(df)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
