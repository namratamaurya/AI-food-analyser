from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import (
    AccuracyCheckRequest,
    AccuracyCheckResponse,
    DailyGoals,
    DailySummary,
    MacroAccuracyMetric,
    MealAnalysisRequest,
    MealAnalysisResponse,
    MacroBreakdown,
)
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
MACRO_FIELDS = ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Nutrition Analyzer backend is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    ai_configured = (
        bool(settings.gemini_api_key)
        if settings.ai_provider == "gemini"
        else bool(settings.openai_api_key)
    )
    return {
        "status": "ok",
        "ai_provider": settings.ai_provider,
        "ai_configured": "true" if ai_configured else "false",
    }


def _daily_summary_response() -> DailySummary:
    summary = storage.get_daily_summary()
    return DailySummary(
        goals=summary["goals"],
        consumed=summary["consumed"],
        remaining=summary["remaining"],
    )


def _build_analysis_response(analysis: dict) -> MealAnalysisResponse:
    return MealAnalysisResponse(
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
        is_fallback=analysis.get("is_fallback", False),
        fallback_reason=analysis.get("fallback_reason"),
    )


def _store_real_analysis(response: MealAnalysisResponse, user_id: str | None = None) -> None:
    if response.is_fallback:
        return

    storage.add_meal(
        {
            "meal_name": response.meal_name,
            "confidence": response.confidence,
            "macros": response.macros.model_dump(),
            "summary": response.summary,
            "ingredients": [ingredient.model_dump() for ingredient in response.ingredients],
            "detected_tags": response.detected_tags,
            "user_id": user_id,
        }
    )


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

    response = _build_analysis_response(analysis)
    _store_real_analysis(response, payload.user_id)
    response.cumulative_summary = _daily_summary_response()
    return response


@app.post("/upload-image", response_model=MealAnalysisResponse)
async def upload_image(file: UploadFile = File(...), notes: str | None = None) -> MealAnalysisResponse:
    image_bytes = await file.read()
    analysis = ai_service.analyze_image(image_bytes, notes, file.content_type or "image/jpeg")
    response = _build_analysis_response(analysis)
    _store_real_analysis(response)
    response.cumulative_summary = _daily_summary_response()
    return response


@app.post("/accuracy", response_model=AccuracyCheckResponse)
def check_accuracy(payload: AccuracyCheckRequest) -> AccuracyCheckResponse:
    metrics: dict[str, MacroAccuracyMetric] = {}
    percent_errors = []
    accuracy_scores = []

    for field in MACRO_FIELDS:
        predicted = float(getattr(payload.predicted_macros, field))
        actual = float(getattr(payload.actual_macros, field))
        absolute_error = abs(predicted - actual)
        percent_error = None
        accuracy_score = None

        if actual == 0:
            if predicted == 0:
                percent_error = 0.0
                accuracy_score = 100.0
        else:
            percent_error = round((absolute_error / abs(actual)) * 100, 2)
            accuracy_score = round(max(0.0, 100.0 - percent_error), 2)

        if percent_error is not None:
            percent_errors.append(percent_error)
        if accuracy_score is not None:
            accuracy_scores.append(accuracy_score)

        metrics[field] = MacroAccuracyMetric(
            predicted=predicted,
            actual=actual,
            absolute_error=round(absolute_error, 2),
            percent_error=percent_error,
            accuracy_score=accuracy_score,
        )

    average_percent_error = round(sum(percent_errors) / len(percent_errors), 2) if percent_errors else None
    overall_accuracy_score = round(sum(accuracy_scores) / len(accuracy_scores), 2) if accuracy_scores else None
    return AccuracyCheckResponse(
        overall_accuracy_score=overall_accuracy_score,
        average_percent_error=average_percent_error,
        metrics=metrics,
    )


@app.post("/goals", response_model=DailyGoals)
def set_goals(payload: DailyGoals) -> DailyGoals:
    return storage.set_goals(payload)


@app.get("/daily-summary", response_model=DailySummary)
def daily_summary() -> DailySummary:
    return _daily_summary_response()


@app.get("/meal-history")
def meal_history() -> list[dict]:
    return storage.get_meal_history()
