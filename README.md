# remove-bg-store-service

Standalone Python service for `/api/studio/remove-bg-store` compatibility.

This service:

- Accepts the same request payload shape as the current Next.js endpoint.
- Calls an upstream local remove-bg worker (`/remove-bg`) for PNG cutout generation.
- Computes `width`, `height`, `stats`, and `objects` from the resulting PNG alpha channel.
- Stores generated cutouts under `generated-assets/external-cutouts`.
- Serves static assets at `/generated-assets/**`.
- Exposes both routes for compatibility:
  - `POST /api/studio/remove-bg-store`
  - `POST /remove-bg-store`

## Repository Layout

- `app/main.py`: FastAPI app entrypoint
- `app/routes/remove_bg_store.py`: endpoint implementation
- `app/services/subject_metrics.py`: alpha-based metrics and object bounds
- `app/services/storage.py`: deterministic filename hashing
- `app/static/generated-assets/external-cutouts`: output storage
- `worker/rembg_service.py`: copied upstream remove-bg worker from the main repo

## 1) Setup

```bash
cd remove-bg-store-service
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy env file:

```bash
copy .env.example .env
```

## 2) Run service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## 3) Run worker (separate process)

Install worker dependencies:

```bash
pip install -r worker/requirements.worker.txt
```

Run:

```bash
python worker/rembg_service.py
```

By default worker listens on `http://127.0.0.1:7860`.

## 4) API Contract

`POST /api/studio/remove-bg-store`

Request body example:

```json
{
  "imageUrl": "https://example.com/photo.jpg",
  "model": "bria-rmbg-2.0",
  "alphaMattingFgThreshold": 200,
  "alphaMattingBgThreshold": 40,
  "alphaMattingErodeSize": 8,
  "edgeFeatherPx": 0.2
}
```

Response shape:

```json
{
  "url": "http://localhost:8000/generated-assets/external-cutouts/cutout-xxxxxxxxxxxxxxxx.png",
  "relativeUrl": "/generated-assets/external-cutouts/cutout-xxxxxxxxxxxxxxxx.png",
  "model": "bria-rmbg-2.0",
  "width": 1024,
  "height": 1536,
  "generationMs": 723,
  "totalMs": 755,
  "stats": {
    "objectCount": 1,
    "coverage": 72.34,
    "opaqueCoverage": 27.66
  },
  "objects": [{ "x": 120, "y": 80, "width": 780, "height": 1400 }]
}
```

## Notes

- The endpoint path can remain `/api/studio/remove-bg-store` behind a reverse proxy.
- This scaffold is response-compatible by contract and field names.
- For strict pixel-perfect parity with the TypeScript implementation, keep adding fixture-based comparison tests between old and new services.
