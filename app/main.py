from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_ROOT, ensure_external_cutout_dir
from app.routes.remove_bg_store import router as remove_bg_store_router


ensure_external_cutout_dir()

app = FastAPI(title="remove-bg-store-service", version="0.1.0")

app.mount(
    "/generated-assets",
    StaticFiles(directory=str(STATIC_ROOT / "generated-assets")),
    name="generated-assets",
)

app.include_router(remove_bg_store_router)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
