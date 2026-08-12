import base64
import json
import struct
from typing import Any
from urllib.parse import urlparse
import zlib

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

        return self._analyze_image_with_openai(image_bytes, notes, mime_type)

    def _analyze_image_with_openai(
        self,
        image_bytes: bytes,
        notes: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        if not self.openai_client:
            return self._fallback_result("OPENAI_API_KEY is not configured.", notes, image_bytes, mime_type)

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
                            "estimated_quantity_g, confidence, macros with calories/protein_g/carbs_g/fat_g/fiber_g, "
                            "and tips with 2-4 practical suggestions for reducing calories or improving this meal."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze the meal image and estimate nutrition. "
                                    "Include practical tips the user can apply to reduce calories or make the meal healthier. "
                                    f"Extra context: {notes or 'No notes provided'}."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                            },
                        ],
                    },
                ],
            )
            raw_text = response.choices[0].message.content or "{}"
            payload = self._parse_json_response(raw_text)
            return self._normalize(payload, notes)
        except OpenAIError as exc:
            return self._fallback_result(f"OpenAI request failed: {exc.__class__.__name__}.", notes, image_bytes, mime_type)
        except json.JSONDecodeError:
            return self._fallback_result("OpenAI returned a response that was not valid JSON.", notes, image_bytes, mime_type)
        except (KeyError, TypeError, ValueError) as exc:
            return self._fallback_result(f"AI response could not be normalized: {exc.__class__.__name__}.", notes, image_bytes, mime_type)

    def _analyze_image_with_gemini(
        self,
        image_bytes: bytes,
        notes: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        if not self.settings.gemini_api_key:
            return self._fallback_result("GEMINI_API_KEY is not configured.", notes, image_bytes, mime_type)

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
                                        "meal_name, confidence, summary, detected_tags, ingredients, macros, and tips. "
                                        "Each ingredient must include name, estimated_quantity_g, confidence, and "
                                        "macros with calories, protein_g, carbs_g, fat_g, fiber_g. "
                                        "Tips must be 2-4 short, practical suggestions for reducing calories or improving this meal. "
                                        f"Extra context: {notes or 'No notes provided'}."
                                    )
                                },
                            ],
                        }
                    ],
                    "generationConfig": {
                        "maxOutputTokens": 4096,
                        "responseMimeType": "application/json",
                        "responseSchema": self._gemini_response_schema(),
                    },
                },
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
            raw_text = self._extract_gemini_text(payload)
            return self._normalize(self._parse_json_response(raw_text), notes)
        except httpx.HTTPStatusError as exc:
            return self._fallback_result(f"Gemini request failed with HTTP {exc.response.status_code}.", notes, image_bytes, mime_type)
        except httpx.HTTPError as exc:
            return self._fallback_result(f"Gemini request failed: {exc.__class__.__name__}.", notes, image_bytes, mime_type)
        except json.JSONDecodeError:
            return self._fallback_result("Gemini returned a response that was not valid JSON.", notes, image_bytes, mime_type)
        except ValueError as exc:
            return self._fallback_result(f"Gemini response could not be normalized: {exc}", notes, image_bytes, mime_type)
        except (KeyError, TypeError) as exc:
            return self._fallback_result(f"Gemini response could not be normalized: {exc.__class__.__name__}.", notes, image_bytes, mime_type)

    def _extract_gemini_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        text_parts = []
        for candidate in candidates:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                text = part.get("text")
                if text:
                    text_parts.append(text)

        if text_parts:
            return "\n".join(text_parts)

        finish_reasons = [
            candidate.get("finishReason")
            for candidate in candidates
            if candidate.get("finishReason")
        ]
        if finish_reasons:
            raise ValueError(f"Gemini returned no text. Finish reason: {', '.join(finish_reasons)}.")
        raise ValueError("Gemini returned no text content.")

    def _parse_json_response(self, raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(cleaned):
                if char not in "{[":
                    continue
                try:
                    payload, _ = decoder.raw_decode(cleaned[index:])
                    break
                except json.JSONDecodeError:
                    continue
            else:
                raise

        if not isinstance(payload, dict):
            raise ValueError("AI response JSON must be an object.")
        return payload

    def _gemini_response_schema(self) -> dict[str, Any]:
        macro_schema = {
            "type": "object",
            "properties": {
                "calories": {"type": "number"},
                "protein_g": {"type": "number"},
                "carbs_g": {"type": "number"},
                "fat_g": {"type": "number"},
                "fiber_g": {"type": "number"},
            },
            "required": ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"],
        }
        return {
            "type": "object",
            "properties": {
                "meal_name": {"type": "string"},
                "confidence": {"type": "number"},
                "summary": {"type": "string"},
                "detected_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "tips": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "estimated_quantity_g": {"type": "number"},
                            "confidence": {"type": "number"},
                            "macros": macro_schema,
                        },
                        "required": ["name", "estimated_quantity_g", "confidence", "macros"],
                    },
                },
                "macros": macro_schema,
            },
            "required": ["meal_name", "confidence", "summary", "detected_tags", "tips", "ingredients", "macros"],
        }

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

    def _fallback_result(
        self,
        reason: str,
        notes: str | None = None,
        image_bytes: bytes | None = None,
        mime_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        visual_estimate = self._visual_fallback_estimate(image_bytes, mime_type)
        if visual_estimate:
            note_text = f" Notes: {notes}" if notes else ""
            visual_reason = f"{reason} Used local visual heuristic fallback."
            return {
                **visual_estimate,
                "summary": (
                    "AI image analysis did not complete, so this is a visual heuristic estimate. "
                    f"Reason: {reason}{note_text} {visual_estimate['summary']}"
                ),
                "detected_tags": ["fallback", "visual-heuristic", *visual_estimate["detected_tags"]],
                "tips": visual_estimate["tips"],
                "is_fallback": True,
                "fallback_reason": visual_reason,
            }

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
            "tips": [
                "Keep portions moderate until a full AI analysis is available.",
                "Add vegetables or lean protein to improve fullness without adding many calories.",
            ],
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

    def _visual_fallback_estimate(self, image_bytes: bytes | None, mime_type: str) -> dict[str, Any] | None:
        if not image_bytes or mime_type != "image/png":
            return None

        try:
            stats = self._png_color_stats(image_bytes)
        except (KeyError, TypeError, ValueError, zlib.error, struct.error):
            return None

        food_area = max(0.01, 1.0 - stats["white"])
        green_food_ratio = stats["green"] / food_area
        fried_food_ratio = stats["fried"] / food_area

        if fried_food_ratio > 0.28 and green_food_ratio < 0.06:
            ingredient_macros = {
                "calories": 780.0,
                "protein_g": 8.0,
                "carbs_g": 92.0,
                "fat_g": 42.0,
                "fiber_g": 7.0,
            }
            return {
                "meal_name": "French fries with sauce",
                "confidence": 0.6,
                "summary": "The image has a strong fried potato and sauce signal, so calories and fat are estimated higher.",
                "detected_tags": ["fried-food", "potato", "sauce", "energy-dense"],
                "tips": [
                    "Choose a smaller portion or share the fries to reduce calories.",
                    "Use less ketchup or serve it on the side to control added sugar.",
                    "Pair with a salad or lean protein so the meal is more filling.",
                ],
                "ingredients": [
                    {
                        "name": "French fries",
                        "estimated_quantity_g": 260.0,
                        "confidence": 0.62,
                        "macros": ingredient_macros,
                    }
                ],
                "macros": ingredient_macros,
            }

        if green_food_ratio > 0.12 or (stats["green"] + stats["purple"] + stats["orange"]) / food_area > 0.24:
            ingredient_macros = {
                "calories": 430.0,
                "protein_g": 14.0,
                "carbs_g": 42.0,
                "fat_g": 24.0,
                "fiber_g": 15.0,
            }
            return {
                "meal_name": "Vegetable salad bowl",
                "confidence": 0.58,
                "summary": "The image has a vegetable-heavy color profile, so fiber is estimated higher and calories lower than fried foods.",
                "detected_tags": ["vegetable-heavy", "salad", "higher-fiber"],
                "tips": [
                    "Keep calorie-dense toppings like seeds and avocado measured rather than free-poured.",
                    "Use a light dressing or lemon-based dressing on the side.",
                    "Add lean protein if you need the bowl to keep you full longer.",
                ],
                "ingredients": [
                    {
                        "name": "Mixed vegetables with toppings",
                        "estimated_quantity_g": 360.0,
                        "confidence": 0.58,
                        "macros": ingredient_macros,
                    }
                ],
                "macros": ingredient_macros,
            }

        return None

    def _png_color_stats(self, image_bytes: bytes) -> dict[str, float]:
        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Image is not a PNG.")

        width = height = color_type = None
        compressed = b""
        position = 8
        while position < len(image_bytes):
            chunk_length = struct.unpack(">I", image_bytes[position : position + 4])[0]
            chunk_type = image_bytes[position + 4 : position + 8]
            chunk_data = image_bytes[position + 8 : position + 8 + chunk_length]
            position += chunk_length + 12

            if chunk_type == b"IHDR":
                width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk_data)
                if bit_depth != 8 or color_type not in {2, 6} or interlace != 0:
                    raise ValueError("Unsupported PNG format.")
            elif chunk_type == b"IDAT":
                compressed += chunk_data
            elif chunk_type == b"IEND":
                break

        if width is None or height is None or color_type is None:
            raise ValueError("PNG header was not found.")

        channels = 4 if color_type == 6 else 3
        bytes_per_pixel = channels
        stride = width * channels
        decompressed = zlib.decompress(compressed)
        rows = []
        previous = bytearray(stride)
        offset = 0

        for _ in range(height):
            filter_type = decompressed[offset]
            offset += 1
            row = bytearray(decompressed[offset : offset + stride])
            offset += stride
            for index, value in enumerate(row):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                up = previous[index]
                upper_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                if filter_type == 1:
                    row[index] = (value + left) & 255
                elif filter_type == 2:
                    row[index] = (value + up) & 255
                elif filter_type == 3:
                    row[index] = (value + ((left + up) // 2)) & 255
                elif filter_type == 4:
                    prediction = self._paeth_predictor(left, up, upper_left)
                    row[index] = (value + prediction) & 255
                elif filter_type != 0:
                    raise ValueError("Unsupported PNG filter.")
            rows.append(row)
            previous = row

        counts = {
            "total": 0,
            "green": 0,
            "fried": 0,
            "red": 0,
            "white": 0,
            "purple": 0,
            "orange": 0,
        }
        step = max(1, min(width, height) // 180)
        for y in range(0, height, step):
            row = rows[y]
            for x in range(0, width, step):
                index = x * channels
                red, green, blue = row[index], row[index + 1], row[index + 2]
                if channels == 4 and row[index + 3] < 20:
                    continue
                counts["total"] += 1
                if green > 70 and green > red * 1.08 and green > blue * 1.08:
                    counts["green"] += 1
                if red > 120 and green > 75 and blue < 125 and red >= green * 0.85:
                    counts["fried"] += 1
                if red > 120 and green < 90 and blue < 90:
                    counts["red"] += 1
                if red > 215 and green > 205 and blue > 185:
                    counts["white"] += 1
                if red > 70 and blue > 70 and red > green * 1.15 and blue > green * 1.05:
                    counts["purple"] += 1
                if red > 150 and 55 < green < 150 and blue < 80:
                    counts["orange"] += 1

        if counts["total"] == 0:
            raise ValueError("PNG contained no visible pixels.")

        return {
            key: value / counts["total"]
            for key, value in counts.items()
            if key != "total"
        }

    def _paeth_predictor(self, left: int, up: int, upper_left: int) -> int:
        estimate = left + up - upper_left
        distance_left = abs(estimate - left)
        distance_up = abs(estimate - up)
        distance_upper_left = abs(estimate - upper_left)
        if distance_left <= distance_up and distance_left <= distance_upper_left:
            return left
        if distance_up <= distance_upper_left:
            return up
        return upper_left

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
        tips = payload.get("tips") or self._default_tips(payload.get("detected_tags", []), macros)
        return {
            "meal_name": payload.get("meal_name", "Detected meal"),
            "confidence": float(payload.get("confidence", 0.7)),
            "summary": payload.get("summary", f"Nutrition estimate for {notes or 'meal'}"),
            "detected_tags": payload.get("detected_tags", []),
            "tips": [str(tip) for tip in tips if str(tip).strip()][:4],
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

    def _default_tips(self, detected_tags: list[str], macros: dict[str, Any]) -> list[str]:
        calories = float(macros.get("calories", 0.0))
        fat = float(macros.get("fat_g", 0.0))
        tags = {str(tag).lower() for tag in detected_tags}
        if "fried-food" in tags or fat >= 25.0:
            return [
                "Reduce fried or oily portions and add vegetables for more volume.",
                "Keep sauces on the side so you can control added calories.",
            ]
        if calories >= 650.0:
            return [
                "Save part of the meal for later or reduce the largest carb or fat portion.",
                "Add water-rich vegetables to make the meal feel larger with fewer calories.",
            ]
        return [
            "Keep dressings, sauces, and calorie-dense toppings measured.",
            "Add lean protein or extra vegetables if you need the meal to be more filling.",
        ]
