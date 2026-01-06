import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/ml_data/delay_data.csv")
X = df.drop("delay_probability", axis=1)
y = df["delay_probability"]

model = RandomForestRegressor(max_depth=4, random_state=42)
model.fit(X, y)

joblib.dump(model, "ml/delay_model.pkl")
