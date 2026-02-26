# Prolix

Prolix is a guided derailment engine: camera → tap → one 250–350 word paragraph that traces an object into destabilizing, honest genealogies.

## Repository layout

- `backend/` FastAPI service implementing `/generate` and `/deepen`, drift engine, retrieval, narration contract, and trace storage.
- `mobile/` Expo React Native app with photo selection, tap-to-generate flow, dark serif UI, and `Go Deeper` / `Return to Surface` controls.

## Backend quickstart

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```



### Backend environment configuration

The backend now supports provider-driven runtime configuration for vision grounding and narration.

Required when using real providers (`openai`):

- `VISION_PROVIDER` (`mock` or `openai`)
- `VISION_MODEL` (for example `gpt-4o-mini`)
- `VISION_API_KEY` (**required** when `VISION_PROVIDER!=mock`)
- `VISION_ENDPOINT` (optional custom base URL)
- `TEXT_PROVIDER` (`mock` or `openai`)
- `TEXT_MODEL` (for example `gpt-4o-mini`)
- `TEXT_API_KEY` (**required** when `TEXT_PROVIDER!=mock`)
- `TEXT_ENDPOINT` (optional custom base URL)
- `AI_REQUEST_TIMEOUT_SECONDS` (default `20`)
- `AI_MAX_RETRIES` (default `2`)

Example:

```bash
export VISION_PROVIDER=openai
export VISION_MODEL=gpt-4o-mini
export VISION_API_KEY=sk-...
export TEXT_PROVIDER=openai
export TEXT_MODEL=gpt-4o-mini
export TEXT_API_KEY=sk-...
```

## Run backend tests

```bash
cd backend
pytest
```

## Mobile quickstart

```bash
cd mobile
npm install
npm run start
```

Or from the repository root:

```bash
npm install
npm run dev
```

Set API endpoint when running on device/simulator:

```bash
EXPO_PUBLIC_API_URL=http://<your-host>:8000 npm run start
```
