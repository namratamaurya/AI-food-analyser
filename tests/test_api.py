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
        return api_module.ai_service._fallback_result("Test fallback.", notes or image_url)

    monkeypatch.setattr(api_module.ai_service, "analyze_image_url", fake_analyze_image_url)

    response = client.post(
        "/analyze-meal",
        json={"image_url": "https://example.com/meal.jpg", "notes": "Lunch"},
    )
    assert response.status_code == 200
    assert response.json()["meal_name"]
    assert response.json()["macros"]["calories"] >= 0
    assert response.json()["is_fallback"] is True
    assert response.json()["fallback_reason"] == "Test fallback."
    assert response.json()["cumulative_summary"]


def test_upload_image_endpoint_accepts_frontend_form_data() -> None:
    response = client.post(
        "/upload-image",
        params={"notes": "Meal photo analysis"},
        files={"file": ("meal.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["meal_name"]
    assert response.json()["detected_tags"]
    assert response.json()["is_fallback"] is True
    assert response.json()["fallback_reason"] == "No image bytes were received by the backend."
    assert response.json()["cumulative_summary"]


def test_upload_image_endpoint_returns_individual_and_cumulative_analysis(monkeypatch) -> None:
    def fake_analyze_image(image_bytes: bytes, notes: str | None = None) -> dict:
        return {
            "meal_name": "Paneer bowl",
            "confidence": 0.9,
            "summary": "Paneer bowl with rice.",
            "detected_tags": ["paneer", "rice"],
            "is_fallback": False,
            "fallback_reason": None,
            "ingredients": [
                {
                    "name": "Paneer",
                    "estimated_quantity_g": 120.0,
                    "confidence": 0.88,
                    "macros": {
                        "calories": 320.0,
                        "protein_g": 22.0,
                        "carbs_g": 8.0,
                        "fat_g": 24.0,
                        "fiber_g": 0.0,
                    },
                }
            ],
            "macros": {
                "calories": 550.0,
                "protein_g": 28.0,
                "carbs_g": 52.0,
                "fat_g": 25.0,
                "fiber_g": 4.0,
            },
        }

    monkeypatch.setattr(api_module.ai_service, "analyze_image", fake_analyze_image)

    response = client.post(
        "/upload-image",
        params={"notes": "Paneer lunch"},
        files={"file": ("meal.jpg", b"image-bytes", "image/jpeg")},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["meal_name"] == "Paneer bowl"
    assert payload["ingredients"][0]["name"] == "Paneer"
    assert payload["is_fallback"] is False
    assert payload["cumulative_summary"]["consumed"]["calories"] >= 550.0

    history_response = client.get("/meal-history")
    assert history_response.status_code == 200
    assert history_response.json()[0]["ingredients"][0]["name"] == "Paneer"


def test_accuracy_endpoint_compares_predicted_and_actual_macros() -> None:
    response = client.post(
        "/accuracy",
        json={
            "predicted_macros": {
                "calories": 720,
                "protein_g": 30,
                "carbs_g": 80,
                "fat_g": 20,
                "fiber_g": 8,
            },
            "actual_macros": {
                "calories": 600,
                "protein_g": 25,
                "carbs_g": 100,
                "fat_g": 25,
                "fiber_g": 10,
            },
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["metrics"]["calories"]["percent_error"] == 20.0
    assert payload["metrics"]["calories"]["accuracy_score"] == 80.0
    assert payload["average_percent_error"] == 20.0
    assert payload["overall_accuracy_score"] == 80.0


def test_accuracy_endpoint_handles_zero_actual_values() -> None:
    response = client.post(
        "/accuracy",
        json={
            "predicted_macros": {
                "calories": 0,
                "protein_g": 5,
                "carbs_g": 0,
                "fat_g": 0,
                "fiber_g": 0,
            },
            "actual_macros": {
                "calories": 0,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
                "fiber_g": 0,
            },
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["metrics"]["calories"]["percent_error"] == 0.0
    assert payload["metrics"]["protein_g"]["percent_error"] is None
    assert payload["overall_accuracy_score"] == 100.0


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
