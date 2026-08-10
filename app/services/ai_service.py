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
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key) if self.settings.openai_api_key else None

    def analyze_image(
        self,
        image_bytes: bytes | None,
        notes: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        if not image_bytes:
            return self._fallback_result("No image bytes were received by the backend.", notes)

        if self.settings.ai_provider == "gemini":
            return self._analyze_image_with_gemini(image_bytes, notes, mime_type)

        if self.settings.ai_provider != "openai":
            return self._fallback_result(f"Unsupported AI_PROVIDER: {self.settings.ai_provider}.", notes)

        return self._analyze_image_with_openai(image_bytes, notes)

    def _analyze_image_with_openai(self, image_bytes: bytes, notes: str | None = None) -> dict[str, Any]:
        if not self.openai_client:
            return self._fallback_result("OPENAI_API_KEY is not configured.", notes)

        try:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            response = self.openai_client.chat.completions.create(
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

    def _analyze_image_with_gemini(
        self,
        image_bytes: bytes,
        notes: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        if not self.settings.gemini_api_key:
            return self._fallback_result("GEMINI_API_KEY is not configured.", notes)

        try:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            response = httpx.post(
                (
                    "https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{self.settings.gemini_model}:generateContent"
                ),
                params={"key": self.settings.gemini_api_key},
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": encoded,
                                    }
                                },
                                {
                                    "text": (
                                        "Analyze this meal image and estimate nutrition. Return JSON only with "
                                        "meal_name, confidence, summary, detected_tags, ingredients, and macros. "
                                        "Each ingredient must include name, estimated_quantity_g, confidence, and "
                                        "macros with calories, protein_g, carbs_g, fat_g, fiber_g. "
                                        f"Extra context: {notes or 'No notes provided'}."
                                    )
                                },
                            ],
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 900,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
            raw_text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return self._normalize(json.loads(raw_text), notes)
        except httpx.HTTPStatusError as exc:
            return self._fallback_result(f"Gemini request failed with HTTP {exc.response.status_code}.", notes)
        except httpx.HTTPError as exc:
            return self._fallback_result(f"Gemini request failed: {exc.__class__.__name__}.", notes)
        except json.JSONDecodeError:
            return self._fallback_result("Gemini returned a response that was not valid JSON.", notes)
        except (KeyError, TypeError, ValueError) as exc:
            return self._fallback_result(f"Gemini response could not be normalized: {exc.__class__.__name__}.", notes)

    def analyze_image_url(self, image_url: str | None, notes: str | None = None) -> dict[str, Any]:
        loaded = self._load_image(image_url)
        return self.analyze_image(loaded["bytes"], notes, loaded["mime_type"])

    def _load_image(self, image_url: str | None) -> dict[str, Any]:
        if not image_url:
            return {"bytes": None, "mime_type": "image/jpeg"}

        if image_url.startswith("data:image/"):
            try:
                header, encoded = image_url.split(",", 1)
                mime_type = header.split(";", 1)[0].replace("data:", "")
                return {"bytes": base64.b64decode(encoded), "mime_type": mime_type}
            except (ValueError, TypeError):
                return {"bytes": None, "mime_type": "image/jpeg"}

        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"}:
            return {"bytes": None, "mime_type": "image/jpeg"}

        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(image_url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    return {"bytes": None, "mime_type": "image/jpeg"}
                return {"bytes": response.content, "mime_type": content_type.split(";", 1)[0]}
        except httpx.HTTPError:
            return {"bytes": None, "mime_type": "image/jpeg"}

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
