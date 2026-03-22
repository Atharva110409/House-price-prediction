# 🏠 House Price Prediction using Machine Learning

## 📌 Overview

This project focuses on predicting house prices using machine learning techniques. It demonstrates a complete ML pipeline — from data preprocessing to model training, evaluation, and prediction.

The goal is to build a reliable regression model that can estimate housing prices based on various input features.

---

## 🚀 Features

* Data preprocessing and cleaning
* Handling missing values
* Feature encoding (categorical → numerical)
* Feature scaling using StandardScaler
* Model training using:

  * Linear Regression
  * Random Forest Regressor
* Model evaluation using R² Score and error metrics
* Automatic selection of best-performing model
* Model saving for future predictions

---

## 🧠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn (for visualization)
* Joblib (for model persistence)

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── data/
│   └── data.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Load dataset
2. Clean and preprocess data
3. Split into training and testing sets
4. Scale features for better performance
5. Train multiple models
6. Evaluate using performance metrics
7. Save the best model

---

## 📊 Model Performance

| Model             | Performance                      |
| ----------------- | -------------------------------- |
| Linear Regression | Baseline model                   |
| Random Forest     | Better accuracy (selected model) |

*(Exact scores may vary depending on dataset)*

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Atharva110409/House-price-prediction.git
cd House-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

---

## 🔮 Making Predictions

The trained model is saved in the `models/` folder.

You can use it via:

```python
from src.predict import predict

result = predict([input_features])
print(result)
```

---

## 📈 Future Improvements

* Add Streamlit web application
* Hyperparameter tuning (GridSearchCV)
* Advanced feature engineering
* Model deployment (cloud)

---

## 💡 Learning Outcomes

Through this project, I learned:

* End-to-end machine learning workflow
* Importance of preprocessing and scaling
* Model comparison techniques
* Writing modular and clean code
* Structuring real-world ML projects

---

## 🤝 Contributing

Feel free to fork this repository and improve it further!

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
