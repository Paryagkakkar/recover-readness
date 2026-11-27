import pandas as pd
import pytest

from app import dashboard


READINGS = {
    "readiness": 8,
    "sleep_duration_h": 7.5,
    "sleep_quality": 4,
    "resting_hr": 58,
    "steps": 8000,
}


def test_dashboard_serves_page_and_falls_back_without_pmdata(tmp_path):
    client = dashboard.create_app(tmp_path).test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"Estimate readiness" in page.data
    assert b'method="post" action="/api/predict"' in page.data
    assert client.get("/static/dashboard.js").status_code == 200
    result = client.post("/api/predict", json=READINGS)

    assert result.status_code == 200
    assert result.get_json() == {"simple_estimate": 8, "model_estimate": None}


@pytest.mark.parametrize(
    ("field", "value"),
    [("readiness", 11), ("sleep_duration_h", float("nan")), ("steps", True),
     ("sleep_quality", 2.5), ("resting_hr", "58")],
)
def test_dashboard_rejects_invalid_readings(tmp_path, field, value):
    client = dashboard.create_app(tmp_path).test_client()
    result = client.post("/api/predict", json={**READINGS, field: value})

    assert result.status_code == 400
    assert field in result.get_json()["errors"]


def test_model_is_loaded_once_and_receives_explicit_feature_order(monkeypatch, tmp_path):
    loads = []

    def train(data_dir):
        loads.append(data_dir)

        class Model:
            def predict(self, data):
                assert list(data.columns) == list(READINGS)
                assert data.iloc[0].to_dict() == READINGS
                return pd.Series([10.5])

        return Model()

    monkeypatch.setattr(dashboard, "train_demo_model", train)
    client = dashboard.create_app(tmp_path).test_client()
    for _ in range(2):
        result = client.post("/api/predict", json=READINGS)
        assert result.get_json() == {"simple_estimate": 8, "model_estimate": 10}

    assert loads == [tmp_path]
