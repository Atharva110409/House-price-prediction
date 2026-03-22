import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Convert categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_features_target(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
