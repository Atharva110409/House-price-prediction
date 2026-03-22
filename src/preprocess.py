import pandas as pd
from datetime import datetime

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df = df.dropna()

    current_year = datetime.now().year
    df['House_Age'] = current_year - df['YearBuilt']

    X = df[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'House_Age']]
    y = df['Price']

    return X, y