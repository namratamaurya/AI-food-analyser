import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_analyze_meal_endpoint() -> None:
    response = client.post(
        "/analyze-meal",
        json={"image_url": "https://example.com/meal.jpg", "notes": "Lunch"},
    )
    assert response.status_code == 200
    assert response.json()["meal_name"]
    assert response.json()["macros"]["calories"] >= 0


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
