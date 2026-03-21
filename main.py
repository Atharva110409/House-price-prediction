import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(r"C:\Users\athar\OneDrive\Desktop\ML_Project\house price prediction\data.csv")


# Remove missing values
df = df.dropna()

# Select only numeric features
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(result.head())


# Evaluate
#print(predictions)
#print(predictions[:5])
#print(mean_absolute_error(y_test, predictions))

my_house =pd.DataFrame([[1500, 4, 3, 3, 2020]],columns=['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt'])
  # Area, Bedrooms, Bathrooms, Floors, YearBuilt

prediction = model.predict(my_house)

print("Predicted Price:", prediction[0])
