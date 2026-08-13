from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId

from app.config import Settings
from app.models import DailyGoals, MacroBreakdown

DEFAULT_USER_ID = "default"


class StorageUnavailableError(RuntimeError):
    pass


class StorageRepository(ABC):
    @abstractmethod
    def set_goals(self, goals: DailyGoals, user_id: str = DEFAULT_USER_ID) -> DailyGoals:
        raise NotImplementedError

    @abstractmethod
    def add_meal(self, meal: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_daily_summary(self, user_id: str = DEFAULT_USER_ID) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_meal_history(self, user_id: str | None = None, months: int | None = None) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_weekly_summary(self, user_id: str = DEFAULT_USER_ID) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_user_profile(self, user_id: str = DEFAULT_USER_ID) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> dict:
        raise NotImplementedError


class InMemoryStorage(StorageRepository):
    def __init__(self) -> None:
        self.goals_by_user: dict[str, DailyGoals] = {DEFAULT_USER_ID: DailyGoals()}
        self.meal_history: list[dict] = []

    def set_goals(self, goals: DailyGoals, user_id: str = DEFAULT_USER_ID) -> DailyGoals:
        self.goals_by_user[user_id] = goals
        return goals

    def add_meal(self, meal: dict) -> dict:
        document = {
            **meal,
            "user_id": meal.get("user_id") or DEFAULT_USER_ID,
            "created_at": meal.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }
        self.meal_history.append(document)
        return document

    def get_daily_summary(self, user_id: str = DEFAULT_USER_ID) -> dict:
        today = datetime.now(timezone.utc).date()
        meals = [
            meal
            for meal in self.meal_history
            if meal.get("user_id", DEFAULT_USER_ID) == user_id and _parse_created_date(meal.get("created_at")) == today
        ]
        consumed = _sum_macros(meals)
        goals = self.goals_by_user.get(user_id, DailyGoals())
        remaining = MacroBreakdown(
            calories=goals.calories - consumed.calories,
            protein_g=goals.protein_g - consumed.protein_g,
            carbs_g=goals.carbs_g - consumed.carbs_g,
            fat_g=goals.fat_g - consumed.fat_g,
            fiber_g=goals.fiber_g - consumed.fiber_g,
        )
        return {
            "goals": goals,
            "consumed": consumed,
            "remaining": remaining,
        }

    def get_meal_history(self, user_id: str | None = None, months: int | None = None) -> list[dict]:
        meals = self.meal_history
        if user_id is not None:
            meals = [meal for meal in meals if meal.get("user_id", DEFAULT_USER_ID) == user_id]
        if months is not None:
            cutoff = _month_cutoff(months)
            meals = [meal for meal in meals if _parse_created_datetime(meal.get("created_at")) >= cutoff]
        return sorted(meals, key=lambda meal: meal.get("created_at", ""), reverse=True)

    def get_weekly_summary(self, user_id: str = DEFAULT_USER_ID) -> dict:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=6)
        start_day = start.date()
        end_day = end.date()
        meals = [
            meal
            for meal in self.meal_history
            if meal.get("user_id", DEFAULT_USER_ID) == user_id
            and start_day <= _parse_created_date(meal.get("created_at")) <= end_day
        ]
        return {
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "consumed": _sum_macros(meals),
            "scan_count": len(meals),
        }

    def get_user_profile(self, user_id: str = DEFAULT_USER_ID) -> dict:
        history = self.get_meal_history(user_id=user_id)
        return {
            "user_id": user_id,
            "goals": self.goals_by_user.get(user_id, DailyGoals()),
            "today": self.get_daily_summary(user_id),
            "week": self.get_weekly_summary(user_id),
            "streaks": _calculate_streaks(history),
            "total_scans": len(history),
        }

    def get_status(self) -> dict:
        return {
            "backend": "memory",
            "configured": False,
            "available": True,
            "detail": "Using in-memory storage. Data will not persist across server restarts.",
        }


class MongoStorage(StorageRepository):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fallback = InMemoryStorage()
        self._client: MongoClient | None = None
        self._db = None
        self._goals_collection = None
        self._meals_collection = None
        self._available = False
        self._last_error: str | None = None

    def _ensure_connection(self) -> bool:
        if self._available:
            return True
        if not self.settings.mongo_uri:
            return False

        try:
            self._client = MongoClient(self.settings.mongo_uri, serverSelectionTimeoutMS=self.settings.mongo_timeout_ms)
            self._db = self._client[self.settings.mongo_db]
            self._goals_collection = self._db["daily_goals"]
            self._meals_collection = self._db["meal_history"]
            self._db.command("ping")
            self._available = True
            self._last_error = None
            return True
        except (PyMongoError, TimeoutError) as exc:
            self._available = False
            self._last_error = f"{exc.__class__.__name__}: {exc}"
            return False

    def _require_connection(self) -> bool:
        if self._ensure_connection():
            return True
        if self.settings.mongo_uri:
            raise StorageUnavailableError(
                "MongoDB is configured but unavailable. "
                f"Last error: {self._last_error or 'connection failed'}"
            )
        return False

    def set_goals(self, goals: DailyGoals, user_id: str = DEFAULT_USER_ID) -> DailyGoals:
        if self._require_connection():
            payload = {"_id": user_id, "user_id": user_id, **goals.model_dump()}
            self._goals_collection.replace_one({"_id": user_id}, payload, upsert=True)
            return goals
        return self._fallback.set_goals(goals, user_id)

    def add_meal(self, meal: dict) -> dict:
        if self._require_connection():
            document = {
                **meal,
                "user_id": meal.get("user_id") or DEFAULT_USER_ID,
                "created_at": meal.get("created_at") or datetime.now(timezone.utc).isoformat(),
            }
            result = self._meals_collection.insert_one(document)
            return self._serialize_document({**document, "_id": result.inserted_id})
        return self._fallback.add_meal(meal)

    def get_daily_summary(self, user_id: str = DEFAULT_USER_ID) -> dict:
        if self._require_connection():
            today = datetime.now(timezone.utc).date()
            goal_doc = self._goals_collection.find_one({"_id": user_id})
            goals = DailyGoals(**{k: v for k, v in goal_doc.items() if k != "_id"}) if goal_doc else DailyGoals()
            meals = [
                meal
                for meal in self._meals_collection.find({"user_id": user_id})
                if _parse_created_date(meal.get("created_at")) == today
            ]
            consumed = _sum_macros(meals)
            remaining = MacroBreakdown(
                calories=goals.calories - consumed.calories,
                protein_g=goals.protein_g - consumed.protein_g,
                carbs_g=goals.carbs_g - consumed.carbs_g,
                fat_g=goals.fat_g - consumed.fat_g,
                fiber_g=goals.fiber_g - consumed.fiber_g,
            )
            return {"goals": goals, "consumed": consumed, "remaining": remaining}
        return self._fallback.get_daily_summary(user_id)

    def get_meal_history(self, user_id: str | None = None, months: int | None = None) -> list[dict]:
        if self._require_connection():
            query = {}
            if user_id is not None:
                query["user_id"] = user_id
            if months is not None:
                query["created_at"] = {"$gte": _month_cutoff(months).isoformat()}
            return [self._serialize_document(meal) for meal in self._meals_collection.find(query).sort("created_at", -1)]
        return self._fallback.get_meal_history(user_id, months)

    def get_weekly_summary(self, user_id: str = DEFAULT_USER_ID) -> dict:
        if self._require_connection():
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=6)
            meals = list(
                self._meals_collection.find(
                    {
                        "user_id": user_id,
                        "created_at": {"$gte": datetime.combine(start.date(), time.min, timezone.utc).isoformat()},
                    }
                )
            )
            return {
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "consumed": _sum_macros(meals),
                "scan_count": len(meals),
            }
        return self._fallback.get_weekly_summary(user_id)

    def get_user_profile(self, user_id: str = DEFAULT_USER_ID) -> dict:
        if self._require_connection():
            history = self.get_meal_history(user_id=user_id)
            daily = self.get_daily_summary(user_id)
            goal_doc = self._goals_collection.find_one({"_id": user_id})
            goals = DailyGoals(**{k: v for k, v in goal_doc.items() if k not in {"_id", "user_id"}}) if goal_doc else DailyGoals()
            return {
                "user_id": user_id,
                "goals": goals,
                "today": daily,
                "week": self.get_weekly_summary(user_id),
                "streaks": _calculate_streaks(history),
                "total_scans": len(history),
            }
        return self._fallback.get_user_profile(user_id)

    def get_status(self) -> dict:
        available = self._ensure_connection()
        if self.settings.mongo_uri:
            return {
                "backend": "mongodb",
                "configured": True,
                "available": available,
                "database": self.settings.mongo_db,
                "detail": "MongoDB connection is available." if available else self._last_error,
            }
        return self._fallback.get_status()

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


def _month_cutoff(months: int) -> datetime:
    days = max(1, months) * 30
    return datetime.now(timezone.utc) - timedelta(days=days)


def _parse_created_datetime(created_at: str | None) -> datetime:
    if not created_at:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_created_date(created_at: str | None) -> date:
    return _parse_created_datetime(created_at).date()


def _sum_macros(meals: list[dict]) -> MacroBreakdown:
    consumed = MacroBreakdown()
    for meal in meals:
        macros = meal.get("macros", {})
        consumed.calories += float(macros.get("calories", 0.0))
        consumed.protein_g += float(macros.get("protein_g", 0.0))
        consumed.carbs_g += float(macros.get("carbs_g", 0.0))
        consumed.fat_g += float(macros.get("fat_g", 0.0))
        consumed.fiber_g += float(macros.get("fiber_g", 0.0))
    return consumed


def _calculate_streaks(meals: list[dict]) -> dict:
    scan_dates = sorted({_parse_created_date(meal.get("created_at")) for meal in meals}, reverse=True)
    if not scan_dates:
        return {"current_days": 0, "longest_days": 0, "last_scan_date": None}

    today = datetime.now(timezone.utc).date()
    current = 0
    expected = today
    if scan_dates[0] == today - timedelta(days=1):
        expected = today - timedelta(days=1)
    elif scan_dates[0] != today:
        expected = date.min

    for scan_date in scan_dates:
        if scan_date == expected:
            current += 1
            expected -= timedelta(days=1)
        elif scan_date < expected:
            break

    longest = 1
    running = 1
    for previous, scan_date in zip(scan_dates, scan_dates[1:]):
        if previous - scan_date == timedelta(days=1):
            running += 1
        else:
            longest = max(longest, running)
            running = 1
    longest = max(longest, running)

    return {
        "current_days": current,
        "longest_days": longest,
        "last_scan_date": scan_dates[0].isoformat(),
    }
