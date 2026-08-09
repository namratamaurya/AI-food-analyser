"""Seed the MongoDB with sample goals and meals for local development."""
import os
from datetime import datetime, timezone

from pymongo import MongoClient


def main():
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "nutrition_app")
    client = MongoClient(uri)
    db = client[db_name]

    goals = {
        "_id": "default",
        "calories": 2000.0,
        "protein_g": 150.0,
        "carbs_g": 220.0,
        "fat_g": 70.0,
        "fiber_g": 30.0,
    }
    db.daily_goals.replace_one({"_id": "default"}, goals, upsert=True)

    meals = [
        {
            "meal_name": "Sample Salad",
            "macros": {"calories": 350.0, "protein_g": 12.0, "carbs_g": 30.0, "fat_g": 18.0, "fiber_g": 6.0},
            "summary": "Mixed greens with protein",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "meal_name": "Sample Chicken Rice",
            "macros": {"calories": 650.0, "protein_g": 45.0, "carbs_g": 70.0, "fat_g": 20.0, "fiber_g": 4.0},
            "summary": "Chicken breast with rice",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    db.meal_history.insert_many(meals)
    print("Seeded database with sample goals and meals.")


if __name__ == "__main__":
    main()
