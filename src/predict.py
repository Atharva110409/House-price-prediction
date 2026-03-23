import joblib
import numpy as np
import pandas as pd


def load_model_and_scaler(model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    """Load the saved model and scaler from disk."""
    try:
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("[INFO] Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"[ERROR] Could not load model/scaler: {e}\n"
            "Make sure you have run main.py first to train and save the model."
        )


def predict(input_features, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    """
    Predict house price for given input features.

    Args:
        input_features (list or dict or pd.DataFrame):
            - list  → raw numeric values in the same order as training features
            - dict  → {feature_name: value, ...}
            - DataFrame → single row DataFrame

    Returns:
        float: Predicted house price
    """
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    # Convert input to numpy array
    if isinstance(input_features, dict):
        input_array = np.array(list(input_features.values())).reshape(1, -1)
    elif isinstance(input_features, pd.DataFrame):
        input_array = input_features.values
    else:
        input_array = np.array(input_features).reshape(1, -1)

    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_array)

    # Predict
    predicted_price = model.predict(input_scaled)[0]
    print(f"\n[PREDICTION] Estimated House Price: ₹{predicted_price:,.2f}")
    return predicted_price


def predict_batch(input_df, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    """
    Predict house prices for multiple rows.

    Args:
        input_df (pd.DataFrame): DataFrame with feature columns (no target column)

    Returns:
        np.ndarray: Array of predicted prices
    """
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    input_scaled  = scaler.transform(input_df.values)
    predictions   = model.predict(input_scaled)

    print(f"[INFO] Predicted {len(predictions)} house prices.")
    return predictions
