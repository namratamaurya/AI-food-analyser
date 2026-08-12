import argparse
import json
import mimetypes
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.services.ai_service import AIService


def food_png_bytes(theme: str, width: int = 128, height: int = 96) -> bytes:
    pixels = bytearray()
    for y in range(height):
        row = bytearray()
        for x in range(width):
            plate_distance = ((x - width / 2) / 54) ** 2 + ((y - height / 2) / 38) ** 2
            rice_distance = ((x - 52) / 24) ** 2 + ((y - 48) / 21) ** 2
            main_distance = ((x - 78) / 23) ** 2 + ((y - 50) / 20) ** 2
            garnish_distance = ((x - 77) / 10) ** 2 + ((y - 37) / 8) ** 2

            color = (158, 117, 82)
            if plate_distance <= 1:
                color = (245, 242, 232)
            if rice_distance <= 1:
                color = (252, 246, 218)
            if main_distance <= 1:
                color = (218, 143, 45) if theme == "dal_rice" else (96, 168, 92)
            if garnish_distance <= 1:
                color = (47, 126, 64)
            row.extend(color)
        pixels.append(0)
        pixels.extend(row)

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(pixels))) + chunk(b"IEND", b"")


def _load_samples(paths: list[str]) -> list[tuple[str, str, bytes, str]]:
    if not paths:
        return [
            ("dal_rice", "A dal and rice bowl with herbs", food_png_bytes("dal_rice"), "image/png"),
            ("salad", "A green salad bowl with grains", food_png_bytes("salad"), "image/png"),
        ]

    samples = []
    for path_text in paths:
        path = Path(path_text).expanduser()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        samples.append((path.stem, path.stem, path.read_bytes(), mime_type))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Run food image analysis smoke tests against the configured AI provider.")
    parser.add_argument("images", nargs="*", help="Optional food image paths. Uses generated samples when omitted.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.ai_provider == "gemini" and not settings.gemini_api_key:
        print("No GEMINI_API_KEY configured. Running local visual fallback check instead.")
    if settings.ai_provider == "openai" and not settings.openai_api_key:
        print("No OPENAI_API_KEY configured. Running local visual fallback check instead.")

    service = AIService(settings)
    samples = _load_samples(args.images)

    for sample_name, notes, image_bytes, mime_type in samples:
        analysis = service.analyze_image(image_bytes, notes, mime_type)
        print(json.dumps({"sample": sample_name, **analysis}, indent=2))
        fallback_reason = analysis.get("fallback_reason") or ""
        is_local_visual_fallback = "Used local visual heuristic fallback" in fallback_reason
        if analysis.get("is_fallback") and not is_local_visual_fallback:
            print(f"Smoke test failed for {sample_name}: {fallback_reason}", file=sys.stderr)
            return 1
        if not analysis.get("meal_name") or float(analysis["macros"].get("calories", 0.0)) <= 0:
            print(f"Smoke test failed for {sample_name}: incomplete nutrition analysis.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
