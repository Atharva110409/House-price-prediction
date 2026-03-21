# House-price-prediction
ML project to predict house prices


# 🏡 House Price Prediction — Minimal ML Pipeline

A clean and minimal implementation of a supervised Machine Learning model to predict house prices using structured data.

This project focuses on **clarity over complexity**, demonstrating how a real-world regression problem can be solved using a simple yet effective pipeline.

---

## ⚙️ Problem Statement

Given features like:
- Area  
- Number of Bedrooms  
- Bathrooms  
- Floors  
- Year Built  

👉 Predict the **price of a house**

---

## 🧠 Approach

Instead of overcomplicating the model, this project follows a **straightforward ML workflow**:

1. Load dataset using Pandas  
2. Remove missing values (`dropna()`)  
3. Select only **relevant numerical features**  
4. Split dataset into training and testing sets  
5. Train a **Linear Regression model**  
6. Evaluate predictions using test data  
7. Predict price for a custom input house  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  


---

## 📊 Model Details

- Algorithm: **Linear Regression**
- Train/Test Split: **80/20**
- Evaluation Metric: *(optional)* Mean Absolute Error

---

## 🔍 Key Implementation Highlights

✔ Removed missing values for clean training  
✔ Used only numeric features to avoid encoding complexity  
✔ Created a comparison table of:
- Actual prices  
- Predicted prices  

✔ Predicted price for a **custom house input**

---

## 🧪 Sample Prediction

```python
my_house = [1500, 4, 3, 3, 2020]

