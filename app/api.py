from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import DailyGoals, DailySummary, MealAnalysisRequest, MealAnalysisResponse, MacroBreakdown
from app.services.ai_service import AIService
from app.services.storage_service import create_storage

settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
storage = create_storage(settings)
ai_service = AIService(settings)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Nutrition Analyzer backend is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "ai_configured": "true" if settings.openai_api_key else "false"}


@app.post("/analyze-meal", response_model=MealAnalysisResponse)
async def analyze_meal(payload: MealAnalysisRequest) -> MealAnalysisResponse:
    analysis = {
        "meal_name": "Sample meal",
        "confidence": 0.92,
        "macros": {
            "calories": 540.0,
            "protein_g": 24.0,
            "carbs_g": 62.0,
            "fat_g": 18.0,
            "fiber_g": 10.0,
        },
        "summary": "A balanced meal estimate generated from the provided context.",
        "ingredients": [],
        "detected_tags": ["balanced"],
    }

    if payload.image_url:
        analysis = ai_service.analyze_image_url(payload.image_url, payload.notes)

    response = MealAnalysisResponse(
        meal_name=analysis["meal_name"],
        confidence=analysis["confidence"],
        macros=MacroBreakdown(**analysis["macros"]),
        summary=analysis["summary"],
        ingredients=[
            {
                "name": ingredient["name"],
                "estimated_quantity_g": ingredient.get("estimated_quantity_g"),
                "confidence": ingredient.get("confidence", 0.0),
                "macros": ingredient["macros"],
            }
            for ingredient in analysis.get("ingredients", [])
        ],
        detected_tags=analysis.get("detected_tags", []),
    )

    storage.add_meal(
        {
            "meal_name": response.meal_name,
            "macros": response.macros.model_dump(),
            "summary": response.summary,
            "user_id": payload.user_id,
        }
    )
    return response


@app.post("/upload-image", response_model=MealAnalysisResponse)
async def upload_image(file: UploadFile = File(...), notes: str | None = None) -> MealAnalysisResponse:
    image_bytes = await file.read()
    analysis = ai_service.analyze_image(image_bytes, notes)
    response = MealAnalysisResponse(
        meal_name=analysis["meal_name"],
        confidence=analysis["confidence"],
        macros=MacroBreakdown(**analysis["macros"]),
        summary=analysis["summary"],
        ingredients=[
            {
                "name": ingredient["name"],
                "estimated_quantity_g": ingredient.get("estimated_quantity_g"),
                "confidence": ingredient.get("confidence", 0.0),
                "macros": ingredient["macros"],
            }
            for ingredient in analysis.get("ingredients", [])
        ],
        detected_tags=analysis.get("detected_tags", []),
    )
    storage.add_meal({"meal_name": response.meal_name, "macros": response.macros.model_dump(), "summary": response.summary})
    return response


@app.post("/goals", response_model=DailyGoals)
def set_goals(payload: DailyGoals) -> DailyGoals:
    return storage.set_goals(payload)


@app.get("/daily-summary", response_model=DailySummary)
def daily_summary() -> DailySummary:
    summary = storage.get_daily_summary()
    return DailySummary(
        goals=summary["goals"],
        consumed=summary["consumed"],
        remaining=summary["remaining"],
    )


@app.get("/meal-history")
def meal_history() -> list[dict]:
    return storage.get_meal_history()
