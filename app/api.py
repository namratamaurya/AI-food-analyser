import base64

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import (
    AccuracyCheckRequest,
    AccuracyCheckResponse,
    DailyGoals,
    DailySummary,
    MealHistoryItem,
    UserHistoryResponse,
    UserProfileResponse,
    MacroAccuracyMetric,
    MealAnalysisRequest,
    MealAnalysisResponse,
    MacroBreakdown,
    PeriodSummary,
    StreakSummary,
)
from app.services.ai_service import AIService
from app.services.storage_service import StorageUnavailableError, create_storage

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
    storage_status = storage.get_status()
    return {
        "status": "ok",
        "ai_provider": settings.ai_provider,
        "ai_configured": "true" if ai_configured else "false",
        "storage_backend": storage_status["backend"],
        "storage_configured": "true" if storage_status["configured"] else "false",
        "storage_available": "true" if storage_status["available"] else "false",
    }


def _daily_summary_response(user_id: str | None = None) -> DailySummary:
    summary = _storage_call(storage.get_daily_summary, user_id or "default")
    return DailySummary(
        goals=summary["goals"],
        consumed=summary["consumed"],
        remaining=summary["remaining"],
    )


def _optional_daily_summary_response(user_id: str | None = None) -> DailySummary | None:
    try:
        return _daily_summary_response(user_id)
    except HTTPException as exc:
        if exc.status_code == 503:
            return None
        raise


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
        tips=analysis.get("tips", []),
        detected_tags=analysis.get("detected_tags", []),
        is_fallback=analysis.get("is_fallback", False),
        fallback_reason=analysis.get("fallback_reason"),
    )


def _store_analysis(
    response: MealAnalysisResponse,
    user_id: str | None = None,
    image_url: str | None = None,
    image_mime_type: str | None = None,
) -> None:
    _storage_call(
        storage.add_meal,
        {
            "meal_name": response.meal_name,
            "confidence": response.confidence,
            "macros": response.macros.model_dump(),
            "summary": response.summary,
            "ingredients": [ingredient.model_dump() for ingredient in response.ingredients],
            "tips": response.tips,
            "detected_tags": response.detected_tags,
            "is_fallback": response.is_fallback,
            "fallback_reason": response.fallback_reason,
            "user_id": user_id,
            "image_url": image_url,
            "image_mime_type": image_mime_type,
        }
    )


def _optional_store_analysis(
    response: MealAnalysisResponse,
    user_id: str | None = None,
    image_url: str | None = None,
    image_mime_type: str | None = None,
) -> None:
    try:
        _store_analysis(response, user_id, image_url, image_mime_type)
    except HTTPException as exc:
        if exc.status_code == 503:
            return
        raise


def _storage_call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except StorageUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _image_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _history_response(user_id: str, months: int = 3) -> UserHistoryResponse:
    meals = _storage_call(storage.get_meal_history, user_id=user_id, months=months)
    total = MacroBreakdown()
    for meal in meals:
        macros = meal.get("macros", {})
        total.calories += float(macros.get("calories", 0.0))
        total.protein_g += float(macros.get("protein_g", 0.0))
        total.carbs_g += float(macros.get("carbs_g", 0.0))
        total.fat_g += float(macros.get("fat_g", 0.0))
        total.fiber_g += float(macros.get("fiber_g", 0.0))

    return UserHistoryResponse(
        user_id=user_id,
        months=months,
        scan_count=len(meals),
        total_macros=total,
        meals=[MealHistoryItem(**meal) for meal in meals],
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
        "tips": [
            "Keep portions aligned with your daily calorie goal.",
            "Add vegetables or lean protein if you need more fullness.",
        ],
        "detected_tags": ["balanced"],
    }

    if payload.image_url:
        analysis = ai_service.analyze_image_url(payload.image_url, payload.notes)

    response = _build_analysis_response(analysis)
    _optional_store_analysis(response, payload.user_id, payload.image_url, "image/url" if payload.image_url else None)
    response.cumulative_summary = _optional_daily_summary_response(payload.user_id)
    return response


@app.post("/upload-image", response_model=MealAnalysisResponse)
async def upload_image(
    file: UploadFile = File(...),
    notes: str | None = None,
    user_id: str | None = None,
) -> MealAnalysisResponse:
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    analysis = ai_service.analyze_image(image_bytes, notes, mime_type)
    response = _build_analysis_response(analysis)
    image_url = _image_data_url(image_bytes, mime_type) if image_bytes else None
    _optional_store_analysis(response, user_id, image_url, mime_type)
    response.cumulative_summary = _optional_daily_summary_response(user_id)
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
def set_goals(payload: DailyGoals, user_id: str | None = None) -> DailyGoals:
    return _storage_call(storage.set_goals, payload, user_id or "default")


@app.get("/daily-summary", response_model=DailySummary)
def daily_summary(user_id: str | None = None) -> DailySummary:
    return _daily_summary_response(user_id)


@app.get("/users/{user_id}/profile", response_model=UserProfileResponse)
def user_profile(user_id: str) -> UserProfileResponse:
    profile = _storage_call(storage.get_user_profile, user_id)
    return UserProfileResponse(
        user_id=profile["user_id"],
        goals=profile["goals"],
        today=DailySummary(**profile["today"]),
        week=PeriodSummary(**profile["week"]),
        streaks=StreakSummary(**profile["streaks"]),
        total_scans=profile["total_scans"],
    )


@app.post("/users/{user_id}/goals", response_model=DailyGoals)
def set_user_goals(user_id: str, payload: DailyGoals) -> DailyGoals:
    return _storage_call(storage.set_goals, payload, user_id)


@app.get("/users/{user_id}/daily-summary", response_model=DailySummary)
def user_daily_summary(user_id: str) -> DailySummary:
    return _daily_summary_response(user_id)


@app.get("/users/{user_id}/weekly-summary", response_model=PeriodSummary)
def user_weekly_summary(user_id: str) -> PeriodSummary:
    return PeriodSummary(**_storage_call(storage.get_weekly_summary, user_id))


@app.get("/users/{user_id}/history", response_model=UserHistoryResponse)
def user_history(user_id: str, months: int = 3) -> UserHistoryResponse:
    return _history_response(user_id, months)


@app.get("/meal-history")
def meal_history(user_id: str | None = None, months: int | None = None) -> list[dict]:
    return _storage_call(storage.get_meal_history, user_id=user_id, months=months)
