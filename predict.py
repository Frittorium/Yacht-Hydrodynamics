import joblib
import numpy as np

MODEL_PATH = "yacht_gbr.pkl"


def load_model(path=MODEL_PATH):
    ckpt = joblib.load(path)
    return ckpt["model"], ckpt["scaler"], ckpt["feature_names"]


def predict(model, scaler, features: dict) -> float:
    """features: dict with keys matching feature_names, values as floats."""
    model_obj, _, feature_names = load_model()
    x = np.array([features[f] for f in feature_names]).reshape(1, -1)
    return float(model_obj.predict(x)[0])


if __name__ == "__main__":
    model, scaler, feature_names = load_model()
    print(f"Features expected: {feature_names}")

    sample = {
        "LongPos":    -2.3,
        "PrisCoeff":   0.568,
        "LengthDisp":  4.78,
        "BeamDraught": 3.99,
        "LengthBeam":  3.17,
        "FroudeNum":   0.35,
    }
    x = np.array([sample[f] for f in feature_names]).reshape(1, -1)
    result = float(model.predict(x)[0])
    print(f"Predicted residuary resistance: {result:.4f}")