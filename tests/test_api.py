import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.api as api_module
from main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Nutrition Analyzer backend is running"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cors_preflight_allows_frontend_origin() -> None:
    response = client.options(
        "/upload-image",
        headers={
            "Origin": "https://namratamaurya.github.io",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://namratamaurya.github.io"


def test_analyze_meal_endpoint(monkeypatch) -> None:
    def fake_analyze_image_url(image_url: str, notes: str | None = None) -> dict:
        return api_module.ai_service._fallback_result(notes or image_url)

    monkeypatch.setattr(api_module.ai_service, "analyze_image_url", fake_analyze_image_url)

    response = client.post(
        "/analyze-meal",
        json={"image_url": "https://example.com/meal.jpg", "notes": "Lunch"},
    )
    assert response.status_code == 200
    assert response.json()["meal_name"]
    assert response.json()["macros"]["calories"] >= 0


def test_upload_image_endpoint_accepts_frontend_form_data() -> None:
    response = client.post(
        "/upload-image",
        params={"notes": "Meal photo analysis"},
        files={"file": ("meal.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["meal_name"]
    assert response.json()["detected_tags"]


def test_goals_and_daily_summary() -> None:
    goals_response = client.post(
        "/goals",
        json={"calories": 2200, "protein_g": 180, "carbs_g": 250, "fat_g": 70, "fiber_g": 35},
    )
    assert goals_response.status_code == 200
    assert goals_response.json()["calories"] == 2200

    summary_response = client.get("/daily-summary")
    assert summary_response.status_code == 200
    assert summary_response.json()["goals"]["calories"] == 2200
