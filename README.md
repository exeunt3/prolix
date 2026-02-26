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

## Web app quickstart

The backend now serves a browser client at `/web` that talks to the same `/generate` and `/deepen` endpoints.

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/web`.

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
