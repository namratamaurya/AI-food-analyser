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


class MealAnalysisResponse(BaseModel):
    meal_name: str
    confidence: float
    macros: MacroBreakdown
    summary: str
    ingredients: list[IngredientAnalysis] = Field(default_factory=list)
    detected_tags: list[str] = Field(default_factory=list)


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
