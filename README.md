# Nutrition Analyzer Backend

This backend provides a FastAPI foundation for an AI-powered nutrition analyzer that can inspect meal images, estimate nutrition, and track daily goals.

## Features

- Image-based meal analysis endpoint
- AI integration with OpenAI Vision-style prompt handling
- Daily nutrition goal tracking
- Meal history storage in memory
- Clean service-oriented structure for future MongoDB integration

## Project structure

- app/api.py: FastAPI routes
- app/models.py: Pydantic request/response models
- app/config.py: Environment-based configuration
- app/services/ai_service.py: AI-powered nutrition analysis logic
- app/services/storage_service.py: In-memory persistence for goals/history

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy the environment file and add your API key:
   ```bash
   cp .env.example .env
   ```
3. Start the server:
   ```bash
   uvicorn app.api:app --reload
   ```

### Local Docker (recommended)

To run MongoDB and the API locally with Docker:

```bash
docker compose up --build
```

This starts a `mongo` service and the API on port `8000`. The compose file configures the API to connect to MongoDB at `mongodb://mongo:27017`.

Environment variables can be set in a `.env` file or exported in your shell. See `.env.example` for defaults.

## API overview

- GET /: health check message
- GET /health: service health and AI configuration status
- POST /analyze-meal: analyze a meal from an image URL and notes
- POST /upload-image: upload an image file for direct analysis, with optional `user_id`; responses include ingredients and calorie-reduction tips
- POST /goals: set daily nutrition goals
- GET /daily-summary: get consumed vs remaining nutrition
- GET /meal-history: list stored meal entries
- GET /users/{user_id}/profile: get user goals, today totals, weekly totals, streaks, and scan count
- POST /users/{user_id}/goals: set nutrition goals for a specific user
- GET /users/{user_id}/daily-summary: get today's calories and macros for a user
- GET /users/{user_id}/weekly-summary: get rolling 7-day calories and macros for a user
- GET /users/{user_id}/history: get the last 3 months of scans by default, including calorie split, ingredients, tips, timestamp, and image reference

## Notes

- If no OpenAI API key is configured, the service uses a deterministic fallback response so development can continue.
- The current implementation uses in-memory storage; MongoDB can be added later via a repository pattern.

## Pre-deploy AI image smoke test

Run this before deploying changes to image analysis:

```bash
AI_PROVIDER=gemini GEMINI_API_KEY=your_key python3 scripts/smoke_test_food_images.py
```

To test specific local images:

```bash
python3 scripts/smoke_test_food_images.py "/path/to/food-1.png" "/path/to/food-2.png"
```

The script generates two local food PNG samples when no paths are provided, or sends the provided files through the configured AI provider. If no provider key is configured, it validates the local visual fallback instead. It exits with a failure if analysis returns a generic fallback or incomplete nutrition values.
