from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

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


class PostgresStorage(StorageRepository):
    def __init__(self, settings: Settings, connection_factory: Callable[..., Any] | None = None) -> None:
        self.settings = settings
        self._fallback = InMemoryStorage()
        self._connection_factory = connection_factory
        self._available = False
        self._initialized = False
        self._last_error: str | None = None

    def _connect(self):
        if not self.settings.postgres_url:
            return None

        if self._connection_factory:
            return self._connection_factory(self.settings.postgres_url)

        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.settings.postgres_url, row_factory=dict_row)

    def _ensure_connection(self) -> bool:
        if not self.settings.postgres_url:
            return False

        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    if not self._initialized:
                        self._create_tables(cursor)
                        self._initialized = True
                connection.commit()
            self._available = True
            self._last_error = None
            return True
        except Exception as exc:
            self._available = False
            self._last_error = f"{exc.__class__.__name__}: {exc}"
            return False

    def _require_connection(self) -> bool:
        if self._ensure_connection():
            return True
        if self.settings.postgres_url:
            raise StorageUnavailableError(
                "Postgres is configured but unavailable. "
                f"Last error: {self._last_error or 'connection failed'}"
            )
        return False

    def _create_tables(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_goals (
                user_id TEXT PRIMARY KEY,
                calories DOUBLE PRECISION NOT NULL,
                protein_g DOUBLE PRECISION NOT NULL,
                carbs_g DOUBLE PRECISION NOT NULL,
                fat_g DOUBLE PRECISION NOT NULL,
                fiber_g DOUBLE PRECISION NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS meal_history (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                meal_name TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                macros JSONB NOT NULL,
                summary TEXT NOT NULL,
                ingredients JSONB NOT NULL DEFAULT '[]'::jsonb,
                tips JSONB NOT NULL DEFAULT '[]'::jsonb,
                detected_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                is_fallback BOOLEAN NOT NULL DEFAULT FALSE,
                fallback_reason TEXT,
                image_url TEXT,
                image_mime_type TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meal_history_user_created ON meal_history (user_id, created_at DESC)")

    def set_goals(self, goals: DailyGoals, user_id: str = DEFAULT_USER_ID) -> DailyGoals:
        if self._require_connection():
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO daily_goals (user_id, calories, protein_g, carbs_g, fat_g, fiber_g, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (user_id) DO UPDATE SET
                            calories = EXCLUDED.calories,
                            protein_g = EXCLUDED.protein_g,
                            carbs_g = EXCLUDED.carbs_g,
                            fat_g = EXCLUDED.fat_g,
                            fiber_g = EXCLUDED.fiber_g,
                            updated_at = NOW()
                        """,
                        (user_id, goals.calories, goals.protein_g, goals.carbs_g, goals.fat_g, goals.fiber_g),
                    )
                connection.commit()
            return goals
        return self._fallback.set_goals(goals, user_id)

    def add_meal(self, meal: dict) -> dict:
        if self._require_connection():
            document = {
                **meal,
                "user_id": meal.get("user_id") or DEFAULT_USER_ID,
                "created_at": meal.get("created_at") or datetime.now(timezone.utc).isoformat(),
            }
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO meal_history (
                            user_id, meal_name, confidence, macros, summary, ingredients, tips,
                            detected_tags, is_fallback, fallback_reason, image_url, image_mime_type, created_at
                        )
                        VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            document["user_id"],
                            document["meal_name"],
                            document["confidence"],
                            _json_dump(document["macros"]),
                            document["summary"],
                            _json_dump(document.get("ingredients", [])),
                            _json_dump(document.get("tips", [])),
                            _json_dump(document.get("detected_tags", [])),
                            document.get("is_fallback", False),
                            document.get("fallback_reason"),
                            document.get("image_url"),
                            document.get("image_mime_type"),
                            _parse_created_datetime(document.get("created_at")),
                        ),
                    )
                    inserted = cursor.fetchone()
                connection.commit()
            return {**document, "id": inserted["id"] if isinstance(inserted, dict) else inserted[0]}
        return self._fallback.add_meal(meal)

    def get_daily_summary(self, user_id: str = DEFAULT_USER_ID) -> dict:
        if self._require_connection():
            today_start = datetime.combine(datetime.now(timezone.utc).date(), time.min, timezone.utc)
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM daily_goals WHERE user_id = %s", (user_id,))
                    goal_row = cursor.fetchone()
                    cursor.execute(
                        "SELECT macros FROM meal_history WHERE user_id = %s AND created_at >= %s",
                        (user_id, today_start),
                    )
                    meals = [{"macros": row["macros"]} for row in cursor.fetchall()]
            goals = _goals_from_row(goal_row) if goal_row else DailyGoals()
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
            clauses = []
            params = []
            if user_id is not None:
                clauses.append("user_id = %s")
                params.append(user_id)
            if months is not None:
                clauses.append("created_at >= %s")
                params.append(_month_cutoff(months))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT * FROM meal_history {where} ORDER BY created_at DESC", tuple(params))
                    return [_meal_from_row(row) for row in cursor.fetchall()]
        return self._fallback.get_meal_history(user_id, months)

    def get_weekly_summary(self, user_id: str = DEFAULT_USER_ID) -> dict:
        if self._require_connection():
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=6)
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT macros FROM meal_history WHERE user_id = %s AND created_at >= %s",
                        (user_id, datetime.combine(start.date(), time.min, timezone.utc)),
                    )
                    meals = [{"macros": row["macros"]} for row in cursor.fetchall()]
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
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM daily_goals WHERE user_id = %s", (user_id,))
                    goal_row = cursor.fetchone()
            return {
                "user_id": user_id,
                "goals": _goals_from_row(goal_row) if goal_row else DailyGoals(),
                "today": daily,
                "week": self.get_weekly_summary(user_id),
                "streaks": _calculate_streaks(history),
                "total_scans": len(history),
            }
        return self._fallback.get_user_profile(user_id)

    def get_status(self) -> dict:
        available = self._ensure_connection()
        if self.settings.postgres_url:
            return {
                "backend": "postgres",
                "configured": True,
                "available": available,
                "detail": "Postgres connection is available." if available else self._last_error,
            }
        return self._fallback.get_status()


def create_storage(settings: Settings) -> StorageRepository:
    if settings.postgres_url:
        return PostgresStorage(settings)
    return MongoStorage(settings)


def _json_dump(value: Any) -> str:
    import json

    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return json.dumps(value)


def _goals_from_row(row: dict) -> DailyGoals:
    return DailyGoals(
        calories=float(row.get("calories", 0.0)),
        protein_g=float(row.get("protein_g", 0.0)),
        carbs_g=float(row.get("carbs_g", 0.0)),
        fat_g=float(row.get("fat_g", 0.0)),
        fiber_g=float(row.get("fiber_g", 0.0)),
    )


def _meal_from_row(row: dict) -> dict:
    created_at = row.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.astimezone(timezone.utc).isoformat()
    return {
        "id": row.get("id"),
        "user_id": row.get("user_id") or DEFAULT_USER_ID,
        "meal_name": row.get("meal_name"),
        "confidence": row.get("confidence", 0.0),
        "macros": row.get("macros") or {},
        "summary": row.get("summary") or "",
        "ingredients": row.get("ingredients") or [],
        "tips": row.get("tips") or [],
        "detected_tags": row.get("detected_tags") or [],
        "is_fallback": row.get("is_fallback", False),
        "fallback_reason": row.get("fallback_reason"),
        "image_url": row.get("image_url"),
        "image_mime_type": row.get("image_mime_type"),
        "created_at": created_at,
    }


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
