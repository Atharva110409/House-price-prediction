# 🏠 House Price Prediction

A Machine Learning project that predicts house prices based on features like area, number of rooms, and house age using regression models.

---

## 🚀 Features

* Linear Regression model
* Random Forest model
* Feature Scaling (StandardScaler)
* Feature Engineering (House Age)
* CLI-based prediction input
* Visualization of predictions

---

## 🛠️ Tech Stack

* Python
* pandas
* scikit-learn
* matplotlib

---

## 📂 Project Structure

```
house-price-prediction/
│
├── data/
│   └── data.csv
├── src/
│   └── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python src/main.py
```

---

## 🧠 Models Used

### Linear Regression

* Simple and interpretable
* Works well for linear relationships

### Random Forest

* Ensemble model using multiple decision trees
* Handles non-linear data better
* Usually provides higher accuracy

---

## 📊 Model Evaluation

| Model             | MAE (Error)  | R² Score     |
| ----------------- | ------------ | ------------ |
| Linear Regression |244587.199919 |-0.0127366375 |
| Random Forest     | 255543.48212 | -0.118546812 |


---

## 📈 Visualization

The project includes a scatter plot comparing:

* Actual Prices vs Predicted Prices

---

## 💡 Example Input

```
Area: 1200
Bedrooms: 3
Bathrooms: 3
Floors: 5
Year Built: 2020
```

---

## 💻 Example Output

```
Predicted Price (Linear Regression):  609231.9920559796
Predicted Price (Random Forest): 464294.01

```

---

## 📌 Future Improvements

* Hyperparameter tuning
* Model saving (joblib)
* Web app deployment (Flask)
* Add real-world dataset

---

⭐ If you like this project, consider giving it a star!
