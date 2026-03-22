import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import load_and_preprocess


def train():
    X, y = load_and_preprocess("data/data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # models
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # train
    lr_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train, y_train)

    # predict
    lr_preds = lr_model.predict(X_test_scaled)
    rf_preds = rf_model.predict(X_test)

    # evaluation
    print("Linear R2:", r2_score(y_test, lr_preds))
    print("RF R2:", r2_score(y_test, rf_preds))

    # SAVE EVERYTHING
    pickle.dump(lr_model, open("models/lr_model.pkl", "wb"))
    pickle.dump(rf_model, open("models/rf_model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    print("✅ Models saved!")