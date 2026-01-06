import joblib
import numbers

risk_model = joblib.load("ml/risk_classifier.pkl")
delay_model = joblib.load("ml/delay_model.pkl")
cost_model = joblib.load("ml/cost_model.pkl")


def _sanitize(inputs, fallback):
    if (
        not isinstance(inputs, list)
        or len(inputs) != len(fallback)
        or not all(isinstance(x, numbers.Number) for x in inputs)
    ):
        return fallback
    return inputs


def run_model(name, inputs):

    if name == "risk_classifier":
        inputs = _sanitize(inputs, [0.7, 18, 1, 0.8])
        pred = risk_model.predict([inputs])[0]
        return ["Low", "Medium", "High"][pred]

    if name == "delay_predictor":
        inputs = _sanitize(inputs, [18, 0.6, 5])
        return round(delay_model.predict([inputs])[0] * 100, 1)

    if name == "cost_overrun_predictor":
        inputs = _sanitize(inputs, [0.7, 1, 0.15])
        return round(cost_model.predict([inputs])[0] * 100, 1)

    return "Invalid model"
