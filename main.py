
from src.preprocess import preprocess
from src.train      import train
from src.predict    import predict


DATA_PATH  = "data/data.csv"   # path to your CSV
TARGET_COL = "price"           # name of the target column (house price)
TEST_SIZE  = 0.2               # 20% data used for testing



def main():
    print("=" * 50)
    print("   HOUSE PRICE PREDICTION — ML PIPELINE")
    print("=" * 50)

    # ── STEP 1: Preprocess ──
    print("\n📦 STEP 1: Preprocessing Data...")
    X_train, X_test, y_train, y_test, feature_names, X_scaled, y = preprocess(
        filepath   = DATA_PATH,
        target_col = TARGET_COL,
        test_size  = TEST_SIZE
    )

    # ── STEP 2: Train + Cross-Validate + Plot ──
    print("\n🤖 STEP 2: Training Models + Cross Validation + Graphs...")
    best_model, best_name = train(
        X_train, X_test, y_train, y_test,
        feature_names, X_scaled, y
    )

    # ── STEP 3: Sample Prediction ──
    print("\n🔮 STEP 3: Sample Prediction...")
    print("[INFO] Using first row of test set as sample input...")
    sample_input = X_test[0]   # already scaled — passed directly
    predicted    = best_model.predict([sample_input])[0]
    actual       = list(y_test)[0]

    print(f"  Actual Price    : ₹{actual:,.2f}")
    print(f"  Predicted Price : ₹{predicted:,.2f}")
    print(f"  Difference      : ₹{abs(actual - predicted):,.2f}")

    print("\n✅ Pipeline complete!")
    print("   → Trained model saved  : models/model.pkl")
    print("   → Scaler saved         : models/scaler.pkl")
    print("   → Graphs saved         : graphs/")
    print("=" * 50)


if __name__ == "__main__":
    main()
