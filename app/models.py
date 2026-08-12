from pydantic import BaseModel, Field


class MacroBreakdown(BaseModel):
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0


class MealAnalysisRequest(BaseModel):
    image_url: str | None = None
    notes: str | None = None
    user_id: str | None = None


class IngredientAnalysis(BaseModel):
    name: str
    estimated_quantity_g: float | None = None
    confidence: float = 0.0
    macros: MacroBreakdown


class DailyGoals(BaseModel):
    calories: float = 2000.0
    protein_g: float = 150.0
    carbs_g: float = 220.0
    fat_g: float = 70.0
    fiber_g: float = 30.0


class DailySummary(BaseModel):
    goals: DailyGoals
    consumed: MacroBreakdown
    remaining: MacroBreakdown


class PeriodSummary(BaseModel):
    start_date: str
    end_date: str
    consumed: MacroBreakdown
    scan_count: int = 0


class StreakSummary(BaseModel):
    current_days: int = 0
    longest_days: int = 0
    last_scan_date: str | None = None


class UserProfileResponse(BaseModel):
    user_id: str
    goals: DailyGoals
    today: DailySummary
    week: PeriodSummary
    streaks: StreakSummary
    total_scans: int = 0


class MealHistoryItem(BaseModel):
    meal_name: str
    confidence: float
    macros: MacroBreakdown
    summary: str
    ingredients: list[IngredientAnalysis] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    detected_tags: list[str] = Field(default_factory=list)
    is_fallback: bool = False
    fallback_reason: str | None = None
    user_id: str | None = None
    created_at: str
    image_url: str | None = None
    image_mime_type: str | None = None


class UserHistoryResponse(BaseModel):
    user_id: str
    months: int
    scan_count: int
    total_macros: MacroBreakdown
    meals: list[MealHistoryItem]


class MealAnalysisResponse(BaseModel):
    meal_name: str
    confidence: float
    macros: MacroBreakdown
    summary: str
    ingredients: list[IngredientAnalysis] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    detected_tags: list[str] = Field(default_factory=list)
    is_fallback: bool = False
    fallback_reason: str | None = None
    cumulative_summary: DailySummary | None = None


class MacroAccuracyMetric(BaseModel):
    predicted: float
    actual: float
    absolute_error: float
    percent_error: float | None = None
    accuracy_score: float | None = None


class AccuracyCheckRequest(BaseModel):
    predicted_macros: MacroBreakdown
    actual_macros: MacroBreakdown


class AccuracyCheckResponse(BaseModel):
    overall_accuracy_score: float | None = None
    average_percent_error: float | None = None
    metrics: dict[str, MacroAccuracyMetric]
