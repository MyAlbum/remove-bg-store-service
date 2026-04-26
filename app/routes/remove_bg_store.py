import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.config import (
    DEFAULT_EXTERNAL_CUTOUT_MODEL,
    DEFAULT_REMOVE_BG_ALPHA_MATTING_BG_THRESHOLD,
    DEFAULT_REMOVE_BG_ALPHA_MATTING_ERODE_SIZE,
    DEFAULT_REMOVE_BG_ALPHA_MATTING_FG_THRESHOLD,
    DEFAULT_REMOVE_BG_EDGE_FEATHER_PX,
    NEXT_PUBLIC_URL,
    REMOVE_BG_LOCAL_URL,
    REMOVE_BG_PUBLIC_BASE_URL,
    ensure_external_cutout_dir,
)
from app.models import RemoveBgStoreRequest
from app.services.storage import build_filename
from app.services.subject_metrics import compute_subject_metrics, compute_subject_objects


router = APIRouter()


def _as_optional_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _public_base_url(request: Request) -> str:
    configured = NEXT_PUBLIC_URL or REMOVE_BG_PUBLIC_BASE_URL
    if configured:
        return configured

    proto = request.headers.get("x-forwarded-proto", "http").split(",", 1)[0]
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost:8000"
    host = host.split(",", 1)[0]
    return f"{proto}://{host}"


@router.post("/api/studio/remove-bg-store")
@router.post("/remove-bg-store")
async def remove_bg_store(body: RemoveBgStoreRequest, request: Request):
    t_start = int(time.time() * 1000)

    image_url = body.imageUrl.strip() if isinstance(body.imageUrl, str) else ""
    if not image_url:
        raise HTTPException(status_code=400, detail="Missing imageUrl")

    model = body.model.strip() if isinstance(body.model, str) and body.model.strip() else DEFAULT_EXTERNAL_CUTOUT_MODEL

    alpha_matting_fg_threshold = (
        _as_optional_number(body.alphaMattingFgThreshold)
        or _as_optional_number(body.NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_FG_THRESHOLD)
        or DEFAULT_REMOVE_BG_ALPHA_MATTING_FG_THRESHOLD
    )
    alpha_matting_bg_threshold = (
        _as_optional_number(body.alphaMattingBgThreshold)
        or _as_optional_number(body.NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_BG_THRESHOLD)
        or DEFAULT_REMOVE_BG_ALPHA_MATTING_BG_THRESHOLD
    )
    alpha_matting_erode_size = (
        _as_optional_number(body.alphaMattingErodeSize)
        or _as_optional_number(body.NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_ERODE_SIZE)
        or DEFAULT_REMOVE_BG_ALPHA_MATTING_ERODE_SIZE
    )
    edge_feather_px = (
        _as_optional_number(body.edgeFeatherPx)
        or _as_optional_number(body.NEXT_PUBLIC_REMOVE_BG_EDGE_FEATHER_PX)
        or DEFAULT_REMOVE_BG_EDGE_FEATHER_PX
    )

    t_upstream_start = int(time.time() * 1000)
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            upstream_res = await client.post(
                f"{REMOVE_BG_LOCAL_URL}/remove-bg",
                json={
                    "imageUrl": image_url,
                    "model": model,
                    "alphaMattingEnabled": True,
                    "alphaMattingFgThreshold": alpha_matting_fg_threshold,
                    "alphaMattingBgThreshold": alpha_matting_bg_threshold,
                    "alphaMattingErodeSize": alpha_matting_erode_size,
                    "edgeFeatherPx": edge_feather_px,
                },
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Local remove-bg service unreachable at "
                f"{REMOVE_BG_LOCAL_URL}/remove-bg: {exc}"
            ),
        ) from exc

    if upstream_res.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Local remove-bg service error: {upstream_res.text}",
        )

    content_type = upstream_res.headers.get("content-type", "")
    if "image/png" not in content_type:
        detail = upstream_res.text
        raise HTTPException(
            status_code=502,
            detail=detail or f"Unexpected response content type from local remove-bg service: {content_type}",
        )

    png_bytes = upstream_res.content
    generation_ms = int(time.time() * 1000) - t_upstream_start

    subject_metrics = compute_subject_metrics(png_bytes)
    objects = compute_subject_objects(png_bytes)

    opaque_coverage_percent = round(subject_metrics.foreground_ratio * 100, 2)
    transparent_coverage_percent = round((1 - subject_metrics.foreground_ratio) * 100, 2)

    file_name = build_filename(
        {
            "imageUrl": image_url,
            "model": model,
            "alphaMattingFgThreshold": alpha_matting_fg_threshold,
            "alphaMattingBgThreshold": alpha_matting_bg_threshold,
            "alphaMattingErodeSize": alpha_matting_erode_size,
            "edgeFeatherPx": edge_feather_px,
        }
    )

    directory = ensure_external_cutout_dir()
    absolute_path = Path(directory) / file_name
    if not absolute_path.exists():
        absolute_path.write_bytes(png_bytes)

    relative_url = f"/generated-assets/external-cutouts/{file_name}"
    url = f"{_public_base_url(request)}{relative_url}"
    total_ms = int(time.time() * 1000) - t_start

    return {
        "url": url,
        "relativeUrl": relative_url,
        "model": model,
        "width": subject_metrics.width,
        "height": subject_metrics.height,
        "generationMs": generation_ms,
        "totalMs": total_ms,
        "stats": {
            "objectCount": len(objects),
            "coverage": transparent_coverage_percent,
            "opaqueCoverage": opaque_coverage_percent,
        },
        "objects": objects,
    }
