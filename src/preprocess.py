import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import datetime
 
 
def load_data(filepath):
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df
 
 
def handle_missing_values(df):
    print(f"\n[INFO] Missing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
 
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  -> Filled '{col}' with median")
 
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  -> Filled '{col}' with mode")
 
    print(f"[INFO] Missing values after cleaning: {df.isnull().sum().sum()}")
    return df
 
 
def remove_outliers(df, target_col):
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    print(f"\n[INFO] Outlier removal: removed {before - len(df)} rows from '{target_col}'")
    return df
 
 
def encode_categorical(df):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols) == 0:
        print("[INFO] No categorical columns to encode.")
        return df
    print(f"\n[INFO] Encoding columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"[INFO] Shape after encoding: {df.shape}")
    return df
 
 
def preprocess(filepath, target_col, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_names, X_scaled, y
    """
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = remove_outliers(df, target_col)
 
    # ── Drop columns that are not useful features ──
    drop_cols = [col for col in [
        'id', 'Id', 'ID',      # ID columns
        'date',                 # date string — too noisy
        'street',               # too many unique address values
        'statezip',             # too many unique values
        'country'               # only one value (USA) — no info
    ] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[INFO] Dropped non-feature columns: {drop_cols}")
 
    # ── Feature Engineering ──
    # House age is more meaningful than raw year built
    if 'yr_built' in df.columns:
        current_year = datetime.datetime.now().year
        df['house_age'] = current_year - df['yr_built']
        df = df.drop(columns=['yr_built'])
        print("[INFO] Created 'house_age' from 'yr_built'")
 
    # Was the house ever renovated? (1 = yes, 0 = no)
    if 'yr_renovated' in df.columns:
        df['renovated'] = (df['yr_renovated'] > 0).astype(int)
        df = df.drop(columns=['yr_renovated'])
        print("[INFO] Created 'renovated' from 'yr_renovated'")
 
    df = encode_categorical(df)
 
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
 
    print(f"\n[INFO] Total features: {len(feature_names)}")
    print(f"[INFO] Target column: '{target_col}'")
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("[INFO] Scaler saved → models/scaler.pkl")
 
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
 
    print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_names, X_scaled, y