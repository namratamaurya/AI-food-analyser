import base64
import json
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import OpenAI, OpenAIError

from app.config import Settings, get_settings
from app.models import IngredientAnalysis, MacroBreakdown


class AIService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key) if self.settings.openai_api_key else None

    def analyze_image(self, image_bytes: bytes | None, notes: str | None = None) -> dict[str, Any]:
        if not image_bytes:
            return self._fallback_result("No image bytes were received by the backend.", notes)

        if not self.client:
            return self._fallback_result("OPENAI_API_KEY is not configured.", notes)

        try:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=900,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a nutrition analysis assistant. Inspect the meal image and return JSON only. "
                            "Provide meal_name, confidence, summary, detected_tags, ingredients with name, "
                            "estimated_quantity_g, confidence, macros with calories/protein_g/carbs_g/fat_g/fiber_g."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze the meal image and estimate nutrition. "
                                    f"Extra context: {notes or 'No notes provided'}."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                            },
                        ],
                    },
                ],
            )
            raw_text = response.choices[0].message.content or "{}"
            payload = json.loads(raw_text)
            return self._normalize(payload, notes)
        except OpenAIError as exc:
            return self._fallback_result(f"OpenAI request failed: {exc.__class__.__name__}.", notes)
        except json.JSONDecodeError:
            return self._fallback_result("OpenAI returned a response that was not valid JSON.", notes)
        except (KeyError, TypeError, ValueError) as exc:
            return self._fallback_result(f"AI response could not be normalized: {exc.__class__.__name__}.", notes)

    def analyze_image_url(self, image_url: str | None, notes: str | None = None) -> dict[str, Any]:
        image_bytes = self._load_image_bytes(image_url)
        return self.analyze_image(image_bytes, notes)

    def _load_image_bytes(self, image_url: str | None) -> bytes | None:
        if not image_url:
            return None

        if image_url.startswith("data:image/"):
            try:
                _, encoded = image_url.split(",", 1)
                return base64.b64decode(encoded)
            except (ValueError, TypeError):
                return None

        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"}:
            return None

        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(image_url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    return None
                return response.content
        except httpx.HTTPError:
            return None

    def _fallback_result(self, reason: str, notes: str | None = None) -> dict[str, Any]:
        ingredient_macros = {
            "calories": 520.0,
            "protein_g": 22.0,
            "carbs_g": 58.0,
            "fat_g": 19.0,
            "fiber_g": 8.0,
        }
        note_text = f" Notes: {notes}" if notes else ""
        return {
            "meal_name": "Detected meal",
            "confidence": 0.72,
            "summary": f"AI image analysis did not complete, so this is a placeholder estimate. Reason: {reason}{note_text}",
            "detected_tags": ["fallback", "image-analysis"],
            "is_fallback": True,
            "fallback_reason": reason,
            "ingredients": [
                {
                    "name": "Mixed meal",
                    "estimated_quantity_g": 400.0,
                    "confidence": 0.72,
                    "macros": ingredient_macros,
                }
            ],
            "macros": ingredient_macros,
        }

    def _normalize(self, payload: dict[str, Any], notes: str | None) -> dict[str, Any]:
        ingredients = payload.get("ingredients") or []
        normalized_ingredients = []
        for ingredient in ingredients:
            macros = ingredient.get("macros") or {}
            normalized_ingredients.append(
                {
                    "name": ingredient.get("name", "Unknown ingredient"),
                    "estimated_quantity_g": ingredient.get("estimated_quantity_g", 0.0),
                    "confidence": float(ingredient.get("confidence", 0.0)),
                    "macros": {
                        "calories": float(macros.get("calories", 0.0)),
                        "protein_g": float(macros.get("protein_g", 0.0)),
                        "carbs_g": float(macros.get("carbs_g", 0.0)),
                        "fat_g": float(macros.get("fat_g", 0.0)),
                        "fiber_g": float(macros.get("fiber_g", 0.0)),
                    },
                }
            )

        macros = payload.get("macros") or {}
        return {
            "meal_name": payload.get("meal_name", "Detected meal"),
            "confidence": float(payload.get("confidence", 0.7)),
            "summary": payload.get("summary", f"Nutrition estimate for {notes or 'meal'}"),
            "detected_tags": payload.get("detected_tags", []),
            "is_fallback": False,
            "fallback_reason": None,
            "ingredients": normalized_ingredients or [
                {
                    "name": "Unknown ingredient",
                    "estimated_quantity_g": 0.0,
                    "confidence": 0.0,
                    "macros": {
                        "calories": 0.0,
                        "protein_g": 0.0,
                        "carbs_g": 0.0,
                        "fat_g": 0.0,
                        "fiber_g": 0.0,
                    },
                }
            ],
            "macros": {
                "calories": float(macros.get("calories", 0.0)),
                "protein_g": float(macros.get("protein_g", 0.0)),
                "carbs_g": float(macros.get("carbs_g", 0.0)),
                "fat_g": float(macros.get("fat_g", 0.0)),
                "fiber_g": float(macros.get("fiber_g", 0.0)),
            },
        }
