from abc import ABC, abstractmethod
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId

from app.config import Settings
from app.models import DailyGoals, MacroBreakdown


class StorageRepository(ABC):
    @abstractmethod
    def set_goals(self, goals: DailyGoals) -> DailyGoals:
        raise NotImplementedError

    @abstractmethod
    def add_meal(self, meal: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_daily_summary(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_meal_history(self) -> list[dict]:
        raise NotImplementedError


class InMemoryStorage(StorageRepository):
    def __init__(self) -> None:
        self.goals = DailyGoals()
        self.consumed = MacroBreakdown()
        self.meal_history: list[dict] = []

    def set_goals(self, goals: DailyGoals) -> DailyGoals:
        self.goals = goals
        return goals

    def add_meal(self, meal: dict) -> dict:
        document = {**meal, "created_at": datetime.now(timezone.utc).isoformat()}
        self.meal_history.append(document)
        self.consumed.calories += meal["macros"]["calories"]
        self.consumed.protein_g += meal["macros"]["protein_g"]
        self.consumed.carbs_g += meal["macros"]["carbs_g"]
        self.consumed.fat_g += meal["macros"]["fat_g"]
        self.consumed.fiber_g += meal["macros"]["fiber_g"]
        return document

    def get_daily_summary(self) -> dict:
        remaining = MacroBreakdown(
            calories=self.goals.calories - self.consumed.calories,
            protein_g=self.goals.protein_g - self.consumed.protein_g,
            carbs_g=self.goals.carbs_g - self.consumed.carbs_g,
            fat_g=self.goals.fat_g - self.consumed.fat_g,
            fiber_g=self.goals.fiber_g - self.consumed.fiber_g,
        )
        return {
            "goals": self.goals,
            "consumed": self.consumed,
            "remaining": remaining,
        }

    def get_meal_history(self) -> list[dict]:
        return self.meal_history


class MongoStorage(StorageRepository):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fallback = InMemoryStorage()
        self._client: MongoClient | None = None
        self._db = None
        self._goals_collection = None
        self._meals_collection = None
        self._available = False

    def _ensure_connection(self) -> bool:
        if self._available:
            return True
        if not self.settings.mongo_uri:
            return False

        try:
            self._client = MongoClient(self.settings.mongo_uri, serverSelectionTimeoutMS=1500)
            self._db = self._client[self.settings.mongo_db]
            self._goals_collection = self._db["daily_goals"]
            self._meals_collection = self._db["meal_history"]
            self._db.command("ping")
            self._available = True
            return True
        except (PyMongoError, TimeoutError):
            self._available = False
            return False

    def set_goals(self, goals: DailyGoals) -> DailyGoals:
        if self._ensure_connection():
            payload = {"_id": "default", **goals.model_dump()}
            self._goals_collection.replace_one({"_id": "default"}, payload, upsert=True)
            return goals
        return self._fallback.set_goals(goals)

    def add_meal(self, meal: dict) -> dict:
        if self._ensure_connection():
            document = {**meal, "created_at": datetime.now(timezone.utc).isoformat()}
            result = self._meals_collection.insert_one(document)
            return self._serialize_document({**document, "_id": result.inserted_id})
        return self._fallback.add_meal(meal)

    def get_daily_summary(self) -> dict:
        if self._ensure_connection():
            goal_doc = self._goals_collection.find_one({"_id": "default"})
            goals = DailyGoals(**{k: v for k, v in goal_doc.items() if k != "_id"}) if goal_doc else DailyGoals()
            meals = list(self._meals_collection.find())
            consumed = MacroBreakdown()
            for meal in meals:
                macros = meal.get("macros", {})
                consumed.calories += float(macros.get("calories", 0.0))
                consumed.protein_g += float(macros.get("protein_g", 0.0))
                consumed.carbs_g += float(macros.get("carbs_g", 0.0))
                consumed.fat_g += float(macros.get("fat_g", 0.0))
                consumed.fiber_g += float(macros.get("fiber_g", 0.0))
            remaining = MacroBreakdown(
                calories=goals.calories - consumed.calories,
                protein_g=goals.protein_g - consumed.protein_g,
                carbs_g=goals.carbs_g - consumed.carbs_g,
                fat_g=goals.fat_g - consumed.fat_g,
                fiber_g=goals.fiber_g - consumed.fiber_g,
            )
            return {"goals": goals, "consumed": consumed, "remaining": remaining}
        return self._fallback.get_daily_summary()

    def get_meal_history(self) -> list[dict]:
        if self._ensure_connection():
            return [self._serialize_document(meal) for meal in self._meals_collection.find().sort("created_at", -1)]
        return self._fallback.get_meal_history()

    def _serialize_document(self, document: dict) -> dict:
        serialized = {}
        for key, value in document.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized


def create_storage(settings: Settings) -> StorageRepository:
    return MongoStorage(settings)
