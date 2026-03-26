"""
AHS Data Loader — Codebook-informed preprocessing

Handles AHS-specific data quirks:
- Binary columns: 1=Yes, 2=No → recode to 1/0
- Sentinel -6 ("Not applicable") → NaN
- BUILT range codes → midpoint years
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path="config/experiment.yaml"):
    """Load experiment configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def recode_binary(df, columns):
    """Recode AHS binary columns: 1=Yes→1, 2=No→0, else→NaN."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 2: 0, "1": 1, "2": 0})
    return df


def handle_na_sentinel(df, sentinel=-6):
    """Replace AHS sentinel values with NaN."""
    df = df.replace(sentinel, np.nan)
    df = df.replace(str(sentinel), np.nan)
    return df


def convert_built_midpoints(df, midpoint_map):
    """Convert BUILT range codes to midpoint years."""
    if "BUILT" not in df.columns or midpoint_map is None:
        return df
    # Convert keys to int for mapping
    midpoint_map = {int(k): v for k, v in midpoint_map.items()}
    df["BUILT"] = df["BUILT"].map(midpoint_map).fillna(df["BUILT"])
    return df


def load_and_preprocess(config):
    """Load both datasets and apply AHS preprocessing."""
    preproc = config.get("preprocessing", {})

    # Load price data
    price_path = config["data"]["price_file"]
    df_price = pd.read_excel(price_path)
    print(f"  Loaded price data: {df_price.shape}")

    # Load insurance data
    insurance_path = config["data"]["insurance_file"]
    df_insurance = pd.read_excel(insurance_path)
    print(f"  Loaded insurance data: {df_insurance.shape}")

    # Apply preprocessing to both
    for df in [df_price, df_insurance]:
        # Handle -6 sentinel
        sentinel = preproc.get("na_sentinel", -6)
        handle_na_sentinel(df, sentinel)

        # Recode binary columns
        binary_cols = preproc.get("binary_columns", [])
        recode_binary(df, binary_cols)

        # Convert BUILT midpoints
        built_map = preproc.get("built_midpoints")
        if built_map:
            convert_built_midpoints(df, built_map)

    # Convert numeric columns
    for df in [df_price, df_insurance]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df_price, df_insurance


def prepare_features(df, feature_cols, target_col):
    """Extract features and target, handling missing values."""
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns (skipped): {missing}")

    # Drop rows where target is missing
    mask = df[target_col].notna()
    df_clean = df[mask].copy()

    X = df_clean[available].copy()
    y = df_clean[target_col].copy()

    # Fill remaining NaN in features with median
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 0)

    return X, y
