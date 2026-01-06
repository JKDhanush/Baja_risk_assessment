import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/ml_data/risk_data.csv")
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

model = RandomForestClassifier(max_depth=4, random_state=42)
model.fit(X, y)

joblib.dump(model, "ml/risk_classifier.pkl")
