import sys
import struct
from datetime import datetime, timedelta, timezone
import zlib
from pathlib import Path

import pytest
import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.api as api_module
from app.config import Settings
from app.services.ai_service import AIService
from main import app

client = TestClient(app)
REAL_FOOD_SCREENSHOTS = [
    Path("/Users/namratamaurya/Desktop/Screenshot 2026-08-12 at 7.45.31 PM.png"),
    Path("/Users/namratamaurya/Desktop/Screenshot 2026-08-12 at 7.46.26 PM.png"),
]
MAKKI_SAAG_SCREENSHOT = Path("/Users/namratamaurya/Desktop/Screenshot 2026-08-13 at 12.27.34 PM.png")


def _food_png_bytes(theme: str = "dal_rice", width: int = 96, height: int = 72) -> bytes:
    pixels = bytearray()
    for y in range(height):
        row = bytearray()
        for x in range(width):
            plate_distance = ((x - width / 2) / 40) ** 2 + ((y - height / 2) / 28) ** 2
            rice_distance = ((x - 40) / 18) ** 2 + ((y - 36) / 16) ** 2
            dal_distance = ((x - 58) / 16) ** 2 + ((y - 38) / 15) ** 2
            garnish_distance = ((x - 57) / 7) ** 2 + ((y - 28) / 5) ** 2

            color = (158, 117, 82)
            if plate_distance <= 1:
                color = (245, 242, 232)
            if rice_distance <= 1:
                color = (252, 246, 218)
            if dal_distance <= 1:
                color = (218, 143, 45) if theme == "dal_rice" else (96, 168, 92)
            if garnish_distance <= 1:
                color = (47, 126, 64)
            row.extend(color)
        pixels.append(0)
        pixels.extend(row)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(pixels))) + chunk(b"IEND", b"")


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Nutrition Analyzer backend is running"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["ai_provider"]


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
    assert response.json()["tips"]
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
    assert response.json()["tips"]
    assert response.json()["cumulative_summary"]


def test_upload_image_endpoint_returns_individual_and_cumulative_analysis(monkeypatch) -> None:
    def fake_analyze_image(image_bytes: bytes, notes: str | None = None, mime_type: str = "image/jpeg") -> dict:
        return {
            "meal_name": "Paneer bowl",
            "confidence": 0.9,
            "summary": "Paneer bowl with rice.",
            "tips": ["Choose grilled paneer or reduce oil to lower calories."],
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
    assert payload["tips"] == ["Choose grilled paneer or reduce oil to lower calories."]
    assert payload["is_fallback"] is False
    assert payload["cumulative_summary"]["consumed"]["calories"] >= 550.0

    history_response = client.get("/meal-history")
    assert history_response.status_code == 200
    assert history_response.json()[0]["ingredients"][0]["name"] == "Paneer"
    assert history_response.json()[0]["tips"] == ["Choose grilled paneer or reduce oil to lower calories."]


def test_upload_image_endpoint_handles_food_png_samples(monkeypatch) -> None:
    seen_uploads = []

    def fake_analyze_image(image_bytes: bytes, notes: str | None = None, mime_type: str = "image/jpeg") -> dict:
        seen_uploads.append((image_bytes, notes, mime_type))
        return {
            "meal_name": notes or "Food sample",
            "confidence": 0.88,
            "summary": "Local food image test analysis.",
            "tips": ["Keep sauces measured."],
            "detected_tags": ["local-test", "food-image"],
            "is_fallback": False,
            "fallback_reason": None,
            "ingredients": [
                {
                    "name": "Sample food",
                    "estimated_quantity_g": 250.0,
                    "confidence": 0.84,
                    "macros": {
                        "calories": 420.0,
                        "protein_g": 16.0,
                        "carbs_g": 58.0,
                        "fat_g": 12.0,
                        "fiber_g": 8.0,
                    },
                }
            ],
            "macros": {
                "calories": 420.0,
                "protein_g": 16.0,
                "carbs_g": 58.0,
                "fat_g": 12.0,
                "fiber_g": 8.0,
            },
        }

    monkeypatch.setattr(api_module.ai_service, "analyze_image", fake_analyze_image)

    for filename, notes, image_bytes in [
        ("dal-rice.png", "Dal rice sample", _food_png_bytes("dal_rice")),
        ("salad-bowl.png", "Salad bowl sample", _food_png_bytes("salad")),
    ]:
        response = client.post(
            "/upload-image",
            params={"notes": notes},
            files={"file": (filename, image_bytes, "image/png")},
        )
        payload = response.json()
        assert response.status_code == 200
        assert payload["meal_name"] == notes
        assert payload["tips"] == ["Keep sauces measured."]
        assert payload["is_fallback"] is False
        assert payload["detected_tags"] == ["local-test", "food-image"]

    assert len(seen_uploads) == 2
    assert all(upload[0].startswith(b"\x89PNG\r\n\x1a\n") for upload in seen_uploads)
    assert all(upload[2] == "image/png" for upload in seen_uploads)


def test_real_food_screenshots_have_different_visual_fallback_estimates() -> None:
    if not all(path.exists() for path in REAL_FOOD_SCREENSHOTS):
        pytest.skip("Real food screenshots are only available on the local development machine.")

    service = AIService(Settings(openai_api_key=None, gemini_api_key=None))
    fries = service.analyze_image(REAL_FOOD_SCREENSHOTS[0].read_bytes(), "Fries screenshot", "image/png")
    salad = service.analyze_image(REAL_FOOD_SCREENSHOTS[1].read_bytes(), "Salad screenshot", "image/png")

    assert fries["meal_name"] == "French fries with sauce"
    assert salad["meal_name"] == "Vegetable salad bowl"
    assert fries["macros"]["calories"] > salad["macros"]["calories"]
    assert fries["macros"]["fat_g"] > salad["macros"]["fat_g"]
    assert salad["macros"]["fiber_g"] > fries["macros"]["fiber_g"]
    assert fries["tips"]
    assert salad["tips"]
    assert "fried-food" in fries["detected_tags"]
    assert "vegetable-heavy" in salad["detected_tags"]


def test_makki_saag_screenshot_visual_fallback_is_not_fries() -> None:
    if not MAKKI_SAAG_SCREENSHOT.exists():
        pytest.skip("Makki saag screenshot is only available on the local development machine.")

    service = AIService(Settings(openai_api_key=None, gemini_api_key=None))
    analysis = service.analyze_image(MAKKI_SAAG_SCREENSHOT.read_bytes(), "Makki saag screenshot", "image/png")

    assert analysis["meal_name"] == "Sarson ka saag with makki roti"
    assert "flatbread" in analysis["detected_tags"]
    assert "fried-food" not in analysis["detected_tags"]
    assert analysis["macros"]["fiber_g"] >= 10.0
    assert analysis["tips"]


def test_gemini_provider_requires_gemini_key() -> None:
    service = AIService(
        Settings(
            ai_provider="gemini",
            openai_api_key=None,
            gemini_api_key=None,
        )
    )

    analysis = service.analyze_image(b"image-bytes", "Dinner", "image/jpeg")
    assert analysis["is_fallback"] is True
    assert analysis["fallback_reason"] == "GEMINI_API_KEY is not configured."


def test_gemini_provider_accepts_markdown_wrapped_json(monkeypatch) -> None:
    class FakeGeminiResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": """```json
{
  "meal_name": "Dal rice",
  "confidence": 0.86,
  "summary": "Dal rice with vegetables.",
  "detected_tags": ["dal", "rice"],
  "tips": ["Use less oil in the dal.", "Add extra vegetables for volume."],
  "ingredients": [
    {
      "name": "Dal",
      "estimated_quantity_g": 180,
      "confidence": 0.82,
      "macros": {
        "calories": 210,
        "protein_g": 12,
        "carbs_g": 30,
        "fat_g": 5,
        "fiber_g": 7
      }
    }
  ],
  "macros": {
    "calories": 520,
    "protein_g": 18,
    "carbs_g": 78,
    "fat_g": 12,
    "fiber_g": 10
  }
}
```"""
                                }
                            ]
                        }
                    }
                ]
            }

    def fake_post(*args, **kwargs) -> FakeGeminiResponse:
        generation_config = kwargs["json"]["generationConfig"]
        assert "temperature" not in generation_config
        assert generation_config["responseMimeType"] == "application/json"
        assert "responseSchema" in generation_config
        return FakeGeminiResponse()

    monkeypatch.setattr("app.services.ai_service.httpx.post", fake_post)
    service = AIService(
        Settings(
            ai_provider="gemini",
            openai_api_key=None,
            gemini_api_key="test-key",
        )
    )

    analysis = service.analyze_image(b"image-bytes", "Dinner", "image/jpeg")
    assert analysis["is_fallback"] is False
    assert analysis["meal_name"] == "Dal rice"
    assert analysis["tips"][0] == "Use less oil in the dal."
    assert analysis["macros"]["calories"] == 520.0


def test_gemini_provider_retries_with_stable_flash_model(monkeypatch) -> None:
    calls = []

    class UnavailableGeminiResponse:
        def raise_for_status(self) -> None:
            request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("unavailable", request=request, response=response)

    class SuccessfulGeminiResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": """{
  "meal_name": "Sarson ka Saag with Makki di Roti",
  "confidence": 0.95,
  "summary": "Mustard greens curry with corn flatbread.",
  "detected_tags": ["indian", "flatbread", "greens"],
  "tips": ["Reduce butter to lower calories."],
  "ingredients": [],
  "macros": {
    "calories": 591,
    "protein_g": 10.2,
    "carbs_g": 66.6,
    "fat_g": 31,
    "fiber_g": 11.3
  }
}"""
                                }
                            ]
                        }
                    }
                ]
            }

    def fake_post(url, *args, **kwargs):
        calls.append(url)
        if "gemini-3.6-flash" in url:
            return UnavailableGeminiResponse()
        return SuccessfulGeminiResponse()

    monkeypatch.setattr("app.services.ai_service.httpx.post", fake_post)
    service = AIService(
        Settings(
            ai_provider="gemini",
            openai_api_key=None,
            gemini_api_key="test-key",
            gemini_model="gemini-3.6-flash",
        )
    )

    analysis = service.analyze_image(b"image-bytes", "Meal photo analysis", "image/png")
    assert analysis["is_fallback"] is False
    assert analysis["meal_name"] == "Sarson ka Saag with Makki di Roti"
    assert any("gemini-3.6-flash" in call for call in calls)
    assert any("gemini-3.5-flash" in call for call in calls)


def test_gemini_provider_accepts_json_from_later_response_part(monkeypatch) -> None:
    class FakeGeminiResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inline_data": {"mime_type": "image/png", "data": "ignored"}},
                                {
                                    "text": """
Here is the estimate:
{
  "meal_name": "Vegetable bowl",
  "confidence": 0.81,
  "summary": "Vegetable bowl with grains.",
  "detected_tags": ["vegetables", "grains"],
  "ingredients": [],
  "macros": {
    "calories": 390,
    "protein_g": 13,
    "carbs_g": 62,
    "fat_g": 9,
    "fiber_g": 11
  }
}
"""
                                },
                            ]
                        }
                    }
                ]
            }

    def fake_post(*args, **kwargs) -> FakeGeminiResponse:
        return FakeGeminiResponse()

    monkeypatch.setattr("app.services.ai_service.httpx.post", fake_post)
    service = AIService(
        Settings(
            ai_provider="gemini",
            openai_api_key=None,
            gemini_api_key="test-key",
        )
    )

    analysis = service.analyze_image(_food_png_bytes("salad"), "Lunch", "image/png")
    assert analysis["is_fallback"] is False
    assert analysis["meal_name"] == "Vegetable bowl"
    assert analysis["macros"]["fiber_g"] == 11.0


def test_gemini_provider_reports_no_text_response(monkeypatch) -> None:
    class FakeGeminiResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}

    def fake_post(*args, **kwargs) -> FakeGeminiResponse:
        return FakeGeminiResponse()

    monkeypatch.setattr("app.services.ai_service.httpx.post", fake_post)
    service = AIService(
        Settings(
            ai_provider="gemini",
            openai_api_key=None,
            gemini_api_key="test-key",
        )
    )

    analysis = service.analyze_image(_food_png_bytes(), "Dinner", "image/png")
    assert analysis["is_fallback"] is True
    assert "Finish reason: SAFETY" in analysis["fallback_reason"]


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


def test_user_profile_tracks_multiple_daily_scans_and_history(monkeypatch) -> None:
    user_id = "profile-test-user"
    analyses = [
        {
            "meal_name": "French fries",
            "confidence": 0.9,
            "summary": "Fries with ketchup.",
            "tips": ["Order a smaller portion next time."],
            "detected_tags": ["fries"],
            "is_fallback": False,
            "fallback_reason": None,
            "ingredients": [
                {
                    "name": "Potato",
                    "estimated_quantity_g": 180.0,
                    "confidence": 0.9,
                    "macros": {
                        "calories": 260.0,
                        "protein_g": 4.0,
                        "carbs_g": 45.0,
                        "fat_g": 8.0,
                        "fiber_g": 4.0,
                    },
                }
            ],
            "macros": {
                "calories": 320.0,
                "protein_g": 5.0,
                "carbs_g": 52.0,
                "fat_g": 10.0,
                "fiber_g": 5.0,
            },
        },
        {
            "meal_name": "Salad bowl",
            "confidence": 0.92,
            "summary": "Vegetable bowl with seeds.",
            "tips": ["Keep seeds measured to manage calories."],
            "detected_tags": ["salad", "healthy"],
            "is_fallback": False,
            "fallback_reason": None,
            "ingredients": [
                {
                    "name": "Mixed vegetables",
                    "estimated_quantity_g": 300.0,
                    "confidence": 0.9,
                    "macros": {
                        "calories": 180.0,
                        "protein_g": 7.0,
                        "carbs_g": 24.0,
                        "fat_g": 7.0,
                        "fiber_g": 10.0,
                    },
                }
            ],
            "macros": {
                "calories": 280.0,
                "protein_g": 10.0,
                "carbs_g": 30.0,
                "fat_g": 14.0,
                "fiber_g": 13.0,
            },
        },
    ]

    def fake_analyze_image(image_bytes: bytes, notes: str | None = None, mime_type: str = "image/jpeg") -> dict:
        return analyses.pop(0)

    monkeypatch.setattr(api_module.ai_service, "analyze_image", fake_analyze_image)

    for name in ["fries.png", "salad.png"]:
        response = client.post(
            "/upload-image",
            params={"notes": name, "user_id": user_id},
            files={"file": (name, _food_png_bytes(), "image/png")},
        )
        assert response.status_code == 200

    profile_response = client.get(f"/users/{user_id}/profile")
    assert profile_response.status_code == 200
    profile = profile_response.json()
    assert profile["today"]["consumed"]["calories"] == 600.0
    assert profile["today"]["consumed"]["fiber_g"] == 18.0
    assert profile["week"]["scan_count"] == 2
    assert profile["week"]["consumed"]["calories"] == 600.0
    assert profile["streaks"]["current_days"] >= 1
    assert profile["total_scans"] == 2

    history_response = client.get(f"/users/{user_id}/history")
    history = history_response.json()
    assert history_response.status_code == 200
    assert history["months"] == 3
    assert history["scan_count"] == 2
    assert history["total_macros"]["calories"] == 600.0
    assert history["meals"][0]["created_at"]
    assert history["meals"][0]["image_url"].startswith("data:image/png;base64,")
    assert history["meals"][0]["ingredients"][0]["name"]
    assert history["meals"][0]["tips"]


def test_user_history_defaults_to_last_three_months() -> None:
    user_id = "history-window-user"
    recent = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    api_module.storage.add_meal(
        {
            "meal_name": "Recent meal",
            "confidence": 0.9,
            "macros": {"calories": 300.0, "protein_g": 12.0, "carbs_g": 40.0, "fat_g": 8.0, "fiber_g": 6.0},
            "summary": "Recent.",
            "ingredients": [],
            "tips": ["Keep portions balanced."],
            "detected_tags": [],
            "is_fallback": False,
            "fallback_reason": None,
            "user_id": user_id,
            "created_at": recent,
            "image_url": "https://example.com/recent.png",
            "image_mime_type": "image/png",
        }
    )
    api_module.storage.add_meal(
        {
            "meal_name": "Old meal",
            "confidence": 0.9,
            "macros": {"calories": 900.0, "protein_g": 20.0, "carbs_g": 100.0, "fat_g": 30.0, "fiber_g": 4.0},
            "summary": "Old.",
            "ingredients": [],
            "tips": ["Keep portions balanced."],
            "detected_tags": [],
            "is_fallback": False,
            "fallback_reason": None,
            "user_id": user_id,
            "created_at": old,
            "image_url": "https://example.com/old.png",
            "image_mime_type": "image/png",
        }
    )

    response = client.get(f"/users/{user_id}/history")
    payload = response.json()
    assert response.status_code == 200
    assert payload["scan_count"] == 1
    assert payload["meals"][0]["meal_name"] == "Recent meal"
    assert payload["total_macros"]["calories"] == 300.0
