🏠 House Price Prediction using Machine Learning

An end-to-end Machine Learning project that predicts house prices using Linear Regression and Random Forest — with cross-validation, feature engineering, and data visualizations.


📌 Overview
This project builds a complete ML pipeline to predict house prices based on features like living area, number of bedrooms, bathrooms, location, and more — using a real-world dataset of 4,600 house sales from Washington, USA.
It covers everything from raw data cleaning to model training, evaluation, cross-validation, and visual analysis — structured in a clean, modular codebase.

🚀 Features

✅ Data loading and exploration
✅ Missing value handling (median / mode fill)
✅ Outlier removal using IQR method
✅ Dropping non-informative columns (date, street, statezip, country)
✅ Feature Engineering:

yr_built → house_age
yr_renovated → renovated (binary)


✅ Categorical encoding (One-Hot Encoding)
✅ Feature scaling using StandardScaler
✅ Model training:

Linear Regression (baseline)
Random Forest Regressor ✅ (best performer)


✅ 5-Fold Cross Validation for reliable scoring
✅ Automatic best model selection
✅ Model evaluation: R², MAE, RMSE
✅ 3 Visualizations saved to /graphs
✅ Model saved with Joblib for reuse
✅ Single and batch prediction support


🧠 Tech Stack
ToolPurposePython 3.xCore languagePandasData manipulationNumPyNumerical operationsScikit-learnML models, scaling, evaluationMatplotlibPlotting graphsSeabornBeautiful visualizationsJoblibModel saving & loading

📂 Project Structure
House-Price-Prediction/
│
├── data/
│   └── data.csv                  # Kaggle House Sales dataset (Washington, USA)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # Cleaning, encoding, feature engineering, scaling
│   ├── train.py                  # Model training, cross-validation, graphs
│   └── predict.py                # Single & batch prediction
│
├── models/
│   ├── model.pkl                 # Saved best model (Random Forest)
│   └── scaler.pkl                # Saved StandardScaler
│
├── graphs/
│   ├── actual_vs_predicted.png   # Graph 1
│   ├── feature_importance.png    # Graph 2
│   └── residuals.png             # Graph 3
│
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
│
├── main.py                       # Entry point — run this
├── requirements.txt
└── README.md

⚙️ How It Works
Raw CSV → Clean → Feature Engineering → Encode → Scale → Train → Evaluate → Predict

Load the dataset from data/data.csv
Clean — fill missing values, remove outliers
Drop useless columns (date, street, statezip, country)
Engineer new features (house_age, renovated)
Encode categorical features (city) using One-Hot Encoding
Scale features with StandardScaler
Train Linear Regression and Random Forest
Cross-validate using 5-Fold CV for reliable R² scores
Auto-select the best performing model
Save model + scaler to models/
Plot 3 graphs and save to graphs/


📊 Model Performance
ModelTest R²CV Mean R²CV Std DevMAERMSELinear Regression0.59250.56040.2358$94,424$142,088Random Forest ✅0.60860.59830.1648$92,847$139,266

Random Forest was automatically selected as the best model based on Test R².
Dataset: 4,600 house sales from Washington, USA (Kaggle). 240 outliers removed via IQR.


📈 Visualizations
1. Actual vs Predicted Prices
Points close to the red dashed line = accurate predictions.
Show Image

2. Feature Importance
Shows which features influence house price the most (Random Forest).
Show Image

3. Residual Distribution
A bell curve centered near 0 means the model errors are balanced.
Show Image

▶️ How to Run
1. Clone the repository
bashgit clone https://github.com/Atharva110409/House-price-prediction.git
cd House-price-prediction
2. Install dependencies
bashpip install -r requirements.txt
3. Run the pipeline
bashpython main.py
This will:

Preprocess the data and engineer features
Train both models with 5-fold cross-validation
Print R², MAE, RMSE in the terminal
Save graphs to graphs/
Save the best model to models/


🔮 Making Predictions
Single prediction:
pythonfrom src.predict import predict

# Feature values in training column order
predicted_price = predict([2000, 3, 2, 2, 0, 1, 800, 0, 1200, 3, 0, 70, 1])
print(predicted_price)
Batch prediction:
pythonimport pandas as pd
from src.predict import predict_batch

df = pd.read_csv("new_houses.csv")
prices = predict_batch(df)
print(prices)

🔁 Cross Validation Output
[INFO] Running 5-Fold Cross Validation for Random Forest...
  CV R² Scores : [0.6966, 0.6521, 0.6982, 0.674, 0.2705]
  Mean R²      : 0.5983
  Std Dev      : 0.1648
The lower score on fold 5 suggests some geographic variation in the city-encoded data — a good candidate for future improvement with better location features.

📦 Requirements
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
Install all with:
bashpip install -r requirements.txt

🔮 Future Improvements

 Hyperparameter tuning with GridSearchCV
 Try XGBoost / LightGBM for better accuracy
 Better location features (lat/long clustering)
 Streamlit web app for interactive predictions
 Deploy to Render or HuggingFace Spaces


💡 What I Learned

End-to-end machine learning workflow on a real dataset
Why cross-validation gives more reliable results than a single train/test split
How feature engineering (house_age, renovated) improves model quality
How to identify and drop non-informative columns
Reading feature importance to understand what drives house prices
Writing modular, clean Python code for ML projects


🤝 Contributing
Feel free to fork this repo, raise issues, or submit pull requests. All contributions are welcome!

⭐ If you found this useful
Give it a star ⭐ on GitHub — it helps a lot!
