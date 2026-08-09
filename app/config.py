import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Nutrition Analyzer API")
    app_version: str = os.getenv("APP_VERSION", "0.2.0")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    mongo_uri: str | None = os.getenv("MONGODB_URI")
    mongo_db: str = os.getenv("MONGODB_DB", "nutrition_app")


def get_settings() -> Settings:
    return Settings()
