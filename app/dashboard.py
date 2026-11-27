"""Serve the local Tabler dashboard and its prediction route."""

from flask import Flask, jsonify, render_template, request

from src.prediction import DATA_DIR, estimate_readiness, train_demo_model


INPUT_RANGES = {
    "readiness": (0, 10, True),
    "sleep_duration_h": (0, 24, False),
    "sleep_quality": (1, 5, True),
    "resting_hr": (20, 220, True),
    "steps": (0, 100_000, True),
}


def create_app(data_dir=DATA_DIR):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 4096
    try:
        model = train_demo_model(data_dir)
    except FileNotFoundError:
        model = None

    @app.get("/")
    def index():
        return render_template("dashboard.html", model_available=model is not None)

    @app.post("/api/predict")
    def predict():
        readings = request.get_json(silent=True)
        if not isinstance(readings, dict):
            return jsonify(message="Send the five readings as a JSON object."), 400

        errors = {}
        for field, (minimum, maximum, whole_number) in INPUT_RANGES.items():
            value = readings.get(field)
            if (
                type(value) not in (int, float)
                or not minimum <= value <= maximum
                or (whole_number and value % 1 != 0)
            ):
                kind = "a whole number" if whole_number else "a number"
                errors[field] = f"Enter {kind} from {minimum:,} to {maximum:,}."
        if errors:
            return jsonify(message="Check the highlighted readings.", errors=errors), 400

        return jsonify(estimate_readiness(readings, model))

    return app
