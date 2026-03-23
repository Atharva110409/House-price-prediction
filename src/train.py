import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


# ─────────────────────────────────────────────
#  GRAPH 1 — Actual vs Predicted
# ─────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='black', linewidths=0.4)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual Price", fontsize=12)
    plt.ylabel("Predicted Price", fontsize=12)
    plt.title(f"Actual vs Predicted — {model_name}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/actual_vs_predicted.png", dpi=150)
    plt.show()
    print("[INFO] Graph saved → graphs/actual_vs_predicted.png")


# ─────────────────────────────────────────────
#  GRAPH 2 — Feature Importance
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        print("[WARN] This model does not support feature importance.")
        return
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    # Fixed: assign hue to avoid FutureWarning
    sns.barplot(x='Importance', y='Feature', data=feat_df,
                hue='Feature', palette='viridis', legend=False)
    plt.title("Top 10 Most Important Features", fontsize=14)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/feature_importance.png", dpi=150)
    plt.show()
    print("[INFO] Graph saved → graphs/feature_importance.png")


# ─────────────────────────────────────────────
#  GRAPH 3 — Residual Distribution
# ─────────────────────────────────────────────
def plot_residuals(y_test, y_pred):
    residuals = np.array(y_test) - np.array(y_pred)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=40, kde=True, color='salmon',
                 edgecolor='black', linewidth=0.4)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.xlabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Residual Distribution", fontsize=14)
    plt.legend()
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/residuals.png", dpi=150)
    plt.show()
    print("[INFO] Graph saved → graphs/residuals.png")


# ─────────────────────────────────────────────
#  CROSS VALIDATION
# ─────────────────────────────────────────────
def run_cross_validation(model, X_scaled, y, model_name, cv=5):
    print(f"\n[INFO] Running {cv}-Fold Cross Validation for {model_name}...")
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    print(f"  CV R² Scores : {[round(float(s), 4) for s in scores]}")
    print(f"  Mean R²      : {scores.mean():.4f}")
    print(f"  Std Dev      : {scores.std():.4f}")
    return scores.mean()


# ─────────────────────────────────────────────
#  EVALUATE MODEL
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{'─'*40}")
    print(f"  Model  : {model_name}")
    print(f"  R²     : {r2:.4f}")
    print(f"  MAE    : {mae:.2f}")
    print(f"  RMSE   : {rmse:.2f}")
    print(f"{'─'*40}")
    return y_pred, r2


# ─────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(X_train, X_test, y_train, y_test, feature_names, X_scaled, y):
    """
    Train Linear Regression and Random Forest.
    Runs cross-validation, evaluates both, saves best model, plots 3 graphs.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_mean = run_cross_validation(model, X_scaled, y, name)
        y_pred, r2_test = evaluate_model(model, X_test, y_test, name)
        results[name] = {
            "model":   model,
            "y_pred":  y_pred,
            "r2_test": r2_test,
            "cv_mean": cv_mean
        }

    # Pick best model by test R²
    best_name = max(results, key=lambda k: results[k]["r2_test"])
    best      = results[best_name]

    print(f"\n[RESULT] Best model → {best_name}")
    print(f"         Test R²   : {best['r2_test']:.4f}")
    print(f"         CV R²     : {best['cv_mean']:.4f}")

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best["model"], "models/model.pkl")
    print(f"[INFO] Model saved → models/model.pkl")

    # Plot all 3 graphs
    plot_actual_vs_predicted(y_test, best["y_pred"], best_name)
    plot_feature_importance(best["model"], feature_names)
    plot_residuals(y_test, best["y_pred"])

    return best["model"], best_name