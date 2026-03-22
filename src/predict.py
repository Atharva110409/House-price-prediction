import pickle
import pandas as pd
from datetime import datetime


def predict():
    # load models
    lr_model = pickle.load(open("models/lr_model.pkl", "rb"))
    rf_model = pickle.load(open("models/rf_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))

    print("\nEnter your house details:")

    area = float(input("Area: "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    floors = int(input("Floors: "))
    year_built = int(input("Year Built: "))

    current_year = datetime.now().year
    house_age = current_year - year_built

    user_data = pd.DataFrame([[area, bedrooms, bathrooms, floors, house_age]],
                             columns=['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'House_Age'])

    # scale for LR
    user_scaled = scaler.transform(user_data)

    # predict
    lr_price = lr_model.predict(user_scaled)
    rf_price = rf_model.predict(user_data)

    print("\n🏠 Linear Regression Price:", lr_price[0])
    print("🌲 Random Forest Price:", rf_price[0])