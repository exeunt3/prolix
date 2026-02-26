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

Set API endpoint when running on device/simulator:

```bash
EXPO_PUBLIC_API_URL=http://<your-host>:8000 npm run start
```
