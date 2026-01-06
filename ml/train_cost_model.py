import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/ml_data/cost_data.csv")
X = df.drop("cost_overrun", axis=1)
y = df["cost_overrun"]

model = RandomForestRegressor(max_depth=4, random_state=42)
model.fit(X, y)

joblib.dump(model, "ml/cost_model.pkl")
