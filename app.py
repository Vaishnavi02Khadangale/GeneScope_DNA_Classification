
from flask import Flask, render_template, jsonify, request
import joblib
import os
import itertools
import numpy as np
import pandas as pd

app = Flask(__name__)

# ---------- LOAD TRAINED MODEL BUNDLE ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dashboard_bundle.joblib")

bundle = joblib.load(MODEL_PATH)

models = bundle["models"]
metrics = bundle["metrics"]
class_names = bundle["class_names"]
roc_curves = bundle.get("roc_curves", {})
roc_auc_scores = bundle.get("roc_auc_scores", {})
feature_columns = list(bundle["feature_columns"])
scaler = bundle.get("scaler")
label_encoder = bundle.get("label_encoder")

DNA_BASES = "ACGT"


# ---------- FIND THE ACTUAL TRAINED MODEL ----------
def extract_estimator(model_data):
    """
    Supports a trained estimator stored directly or inside a dictionary.
    """
    if hasattr(model_data, "predict"):
        return model_data

    if isinstance(model_data, dict):
        # Check common estimator keys first.
        preferred_keys = (
            "model",
            "estimator",
            "classifier",
            "best_estimator",
            "best_model",
            "trained_model",
            "pipeline"
        )

        for key in preferred_keys:
            if key in model_data:
                try:
                    return extract_estimator(model_data[key])
                except ValueError:
                    pass

        # Search remaining nested values.
        for value in model_data.values():
            try:
                return extract_estimator(value)
            except ValueError:
                pass

    raise ValueError(
        "No trained estimator with a predict() method was found "
        "inside the selected model entry."
    )


def find_scaler(model_data):
    """Find a scaler stored inside a model's dictionary, if available."""
    if isinstance(model_data, dict):
        for key in ("scaler", "standard_scaler", "feature_scaler"):
            value = model_data.get(key)
            if value is not None and hasattr(value, "transform"):
                return value

        for value in model_data.values():
            found = find_scaler(value)
            if found is not None:
                return found

    return None


def model_needs_scaling(model_name):
    """Fallback for bundles that store one shared scaler."""
    name = model_name.lower()

    return any(term in name for term in (
        "naive bayes",
        "logistic regression",
        "knn",
        "k-nearest",
        "svm",
        "support vector"
    ))


# ---------- DNA FEATURE EXTRACTION ----------
def create_features(sequence):
    sequence = sequence.upper().strip()

    if not sequence:
        raise ValueError("Please enter a DNA sequence.")

    if not all(base in DNA_BASES for base in sequence):
        raise ValueError("Use only the DNA bases A, C, G and T.")

    if len(sequence) < 3:
        raise ValueError("Enter a DNA sequence containing at least 3 bases.")

    features = {}

    # GC content, represented as a percentage
    features["GC_Content"] = (
        (sequence.count("G") + sequence.count("C"))
        / len(sequence) * 100
    )

    # Retained from the current implementation.
    # This formula must match the formula used during training.
    distinct_3mers = {
        sequence[i:i + 3]
        for i in range(len(sequence) - 2)
    }
    features["kmer_3_freq"] = len(distinct_3mers) / 64

    # Frequencies of all possible 2-mers and 3-mers
    for k in (2, 3):
        all_kmers = [
            "".join(chars)
            for chars in itertools.product(DNA_BASES, repeat=k)
        ]

        total = len(sequence) - k + 1

        for kmer in all_kmers:
            count = sum(
                sequence[i:i + k] == kmer
                for i in range(total)
            )
            features[f"{k}mer_{kmer}"] = count / total

    missing = [
        column for column in feature_columns
        if column not in features
    ]

    if missing:
        raise ValueError(
            "Required features are missing: " + ", ".join(missing)
        )

    # Preserve the exact feature order expected by the trained model.
    return pd.DataFrame(
        [[features[column] for column in feature_columns]],
        columns=feature_columns
    )


# ---------- DASHBOARD ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "project": "GeneScope AI",
        "message": "Backend and trained models loaded successfully"
    })


@app.route("/api/dashboard")
def dashboard_data():
    return jsonify({
        "models": list(models.keys()),
        "metrics": metrics,
        "class_names": list(class_names),
        "roc_curves": roc_curves,
        "roc_auc_scores": roc_auc_scores
    })


# ---------- DNA PREDICTION ----------
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}

        sequence = data.get("sequence", "")
        model_name = data.get("model", "")

        if model_name not in models:
            return jsonify({
                "error": "Please select a valid trained model."
            }), 400

        # Build the feature row.
        X = create_features(sequence)

        # IMPORTANT: extract the estimator from the dictionary.
        model_data = models[model_name]
        model = extract_estimator(model_data)

        # Prefer a scaler stored with this specific model.
        model_scaler = find_scaler(model_data)

        # Use the shared scaler only when needed and available.
        if model_scaler is None and model_needs_scaling(model_name):
            model_scaler = scaler

        # Avoid scaling twice if the estimator is already a pipeline
        # that includes its own preprocessing.
        if model_scaler is not None and hasattr(model, "named_steps"):
            X_input = X
        elif model_scaler is not None:
            X_input = model_scaler.transform(X)
        elif model_needs_scaling(model_name):
            raise ValueError(
                "This model appears to require scaled features, "
                "but no compatible scaler was found in the bundle."
            )
        else:
            X_input = X

        prediction = model.predict(X_input)[0]

        # Convert encoded class predictions back to their original labels.
        if label_encoder is not None:
            encoder_classes = getattr(label_encoder, "classes_", [])
            if isinstance(prediction, (int, np.integer)):
                if 0 <= int(prediction) < len(encoder_classes):
                    prediction = label_encoder.inverse_transform(
                        np.array([int(prediction)])
                    )[0]

        return jsonify({
            "predicted_class": str(prediction),
            "model": model_name,
            "sequence_length": len(sequence.strip())
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("DNA prediction failed")
        return jsonify({
            "error": (
                "Prediction failed. Check the model's feature format "
                "and preprocessing in the trained bundle."
            )
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
