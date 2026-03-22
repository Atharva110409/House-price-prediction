
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import load_data, preprocess_data, split_features_target, scale_features


def train_model(data_path, target_column):
    df = load_data(data_path)
    df = preprocess_data(df)

    X, y = split_features_target(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Models
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predict
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
    print("Random Forest R2:", r2_score(y_test, y_pred_rf))

    # Save best model
    best_model = rf if r2_score(y_test, y_pred_rf) > r2_score(y_test, y_pred_lr) else lr
    joblib.dump(best_model, "models/model.pkl")

    print("Model saved successfully!")
