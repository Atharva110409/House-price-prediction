
import joblib
import numpy as np


def load_model(path="models/model.pkl"):
    return joblib.load(r"C:\Users\athar\OneDrive\Desktop\ML_Project\house price prediction\data\data.csv")


def predict(input_data):
    model = load_model()
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]
