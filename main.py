import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# Load dataset
df = pd.read_csv(r"C:\Users\athar\OneDrive\Desktop\ML_Project\house price prediction\data\data.csv")


# Data Preprocessing
df = df.dropna()

# Feature Engineering
current_year = datetime.now().year
df['House_Age'] = current_year - df['YearBuilt']

X = df[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'House_Age']]
y = df['Price']


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling (only for Linear Regression)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)   # RF does NOT need scaling


# Predictions
lr_preds = lr_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test)


# Evaluation

print("\n📊 Model Performance:\n")

print("Linear Regression:")
print("MAE:", mean_absolute_error(y_test, lr_preds))
print("R2:", r2_score(y_test, lr_preds))

print("\nRandom Forest:")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("R2:", r2_score(y_test, rf_preds))


# Visualization

plt.figure()
plt.scatter(y_test, rf_preds)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted")
plt.show()


# CLI Input

print("\nEnter your house details:")

area = float(input("Area: "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
floors = int(input("Floors: "))
year_built = int(input("Year Built: "))

house_age = current_year - year_built

user_data = pd.DataFrame([[area, bedrooms, bathrooms, floors, house_age]],
                         columns=['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'House_Age'])

# Scale for LR
user_scaled = scaler.transform(user_data)

# Predictions
lr_price = lr_model.predict(user_scaled)
rf_price = rf_model.predict(user_data)

print("\n🏠 Predicted Price (Linear Regression):", lr_price[0])
print("🌲 Predicted Price (Random Forest):", rf_price[0])
