"""
BRIA background removal and LaMa inpainting service.

Uses `rembg` with the `bria-rmbg` model to remove backgrounds from images.
Uses LaMa to fill removed regions during inpainting.

Usage:
    pip install -r requirements.txt
    python birefnet_service.py          # starts on port 7860
    PORT=8000 python birefnet_service.py

The Next.js app talks to this via REMOVE_BG_PROVIDER=local in your .env.local.
Set REMOVE_BG_LOCAL_URL=http://127.0.0.1:7860 (or omit; that's the default).
"""

import asyncio
import gc
import io
import os
import time
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageFilter
from pydantic import BaseModel
import onnxruntime as ort
from rembg import new_session, remove

DEFAULT_MODEL = "ben2"
BRIA_14_MODEL = "bria-rmbg"
BRIA_20_MODEL = "bria-rmbg-2.0"
BEN2_MODEL = "ben2"
U2NET_MODEL = "u2net"
MODNET_MODEL = "modnet"
WITHOUTBG_MODEL = "withoutbg"

# MODNet and withoutbg are exposed as user-facing model choices and mapped to
# robust local rembg sessions for now.
REMBG_MODEL_ALIASES = {
    BRIA_14_MODEL: BRIA_14_MODEL,
    U2NET_MODEL: U2NET_MODEL,
    MODNET_MODEL: "u2net_human_seg",
    WITHOUTBG_MODEL: "birefnet-general",
}

SUPPORTED_REMOVE_BG_MODELS = (
    BRIA_14_MODEL,
    BRIA_20_MODEL,
    BEN2_MODEL,
    U2NET_MODEL,
    MODNET_MODEL,
    WITHOUTBG_MODEL,
)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_INITIAL_ENV_KEYS = set(os.environ.keys())


def _load_env_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if not key or key in _INITIAL_ENV_KEYS:
                    continue

                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                os.environ[key] = value
    except FileNotFoundError:
        pass


_load_env_file(os.path.join(PROJECT_ROOT, ".env"))
_load_env_file(os.path.join(PROJECT_ROOT, ".env.local"))


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.lower() in ("true", "1", "yes", "on")

# ── Image resizing config ────────────────────────────────────────────────────
ENABLE_IMAGE_RESIZING = _get_env_bool("ENABLE_IMAGE_RESIZING", False)
LAMA_FORCE_CPU = _get_env_bool("LAMA_FORCE_CPU", False)
LAMA_INFERENCE_TIMEOUT_SEC = _get_env_float("LAMA_INFERENCE_TIMEOUT_SEC", 90.0)
# LaMa cost scales badly with resolution; cap long edge for the fill step (cutout stays full-res).
LAMA_MAX_LONG_EDGE = _get_env_int("LAMA_MAX_LONG_EDGE", 1024)

# ── Remove-bg defaults ────────────────────────────────────────────────────────
DEFAULT_ALPHA_MATTING_ENABLED = True
DEFAULT_ALPHA_MATTING_FG_THRESHOLD = _get_env_int(
    "NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_FG_THRESHOLD",
    200,
)
DEFAULT_ALPHA_MATTING_BG_THRESHOLD = _get_env_int(
    "NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_BG_THRESHOLD",
    10,
)
DEFAULT_ALPHA_MATTING_ERODE_SIZE = _get_env_int(
    "NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_ERODE_SIZE",
    20,
)
DEFAULT_EDGE_FEATHER_PX = _get_env_float(
    "NEXT_PUBLIC_REMOVE_BG_EDGE_FEATHER_PX",
    1.4,
)
DEFAULT_MASK_HARDNESS = 0.0

# ── Session cache (model name → rembg session) ────────────────────────────────
# Models are loaded on first use and then kept in memory.
_sessions: dict = {}
_ben2: object | None = None
_rmbg2: dict | None = None
_onnx_providers_override: list[str] | None = None
_onnx_cpu_only_models: set[str] = set()


def _get_preferred_providers(model_name: str | None = None) -> list[str]:
    if _onnx_providers_override is not None:
        return _onnx_providers_override

    if model_name is not None and model_name in _onnx_cpu_only_models:
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    # Prefer GPU providers in order: CUDA → DirectML (Windows) → CPU
    for gpu_provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if gpu_provider in available:
            return [gpu_provider, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _set_cpu_only_onnx_mode() -> None:
    global _onnx_providers_override
    _onnx_providers_override = ["CPUExecutionProvider"]


def _reset_onnx_sessions() -> None:
    _sessions.clear()
    gc.collect()


def _reset_onnx_session(model_name: str) -> None:
    _sessions.pop(model_name, None)
    gc.collect()


def _is_onnx_gpu_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    signatures = (
        "dmlfusednode",
        "dmlexecutionprovider",
        "cudaexecutionprovider",
        "cuda",
    )
    return any(sig in message for sig in signatures)


def _is_onnx_device_lost_error(exc: Exception) -> bool:
    message = str(exc).lower()
    signatures = (
        "device instance has been suspended",
        "device removed",
        "887a0005",
    )
    return any(sig in message for sig in signatures)


def _is_onnx_out_of_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    signatures = (
        "8007000e",
        "not enough memory resources are available",
        "out of memory",
    )
    return any(sig in message for sig in signatures)


def _run_rembg_inference_with_recovery(
    infer_image: Image.Image,
    model_name: str,
    alpha_matting_enabled: bool,
    alpha_matting_fg_threshold: int,
    alpha_matting_bg_threshold: int,
    alpha_matting_erode_size: int,
) -> Image.Image:
    resolved_model_name = _resolve_rembg_model_name(model_name)

    def _infer_once() -> Image.Image:
        rembg_session = get_session(resolved_model_name)
        remove_kwargs: dict = {"session": rembg_session}
        if alpha_matting_enabled:
            remove_kwargs["alpha_matting"] = True
            remove_kwargs["alpha_matting_foreground_threshold"] = alpha_matting_fg_threshold
            remove_kwargs["alpha_matting_background_threshold"] = alpha_matting_bg_threshold
            remove_kwargs["alpha_matting_erode_size"] = alpha_matting_erode_size
        return remove(infer_image, **remove_kwargs)

    try:
        return _infer_once()
    except Exception as exc:
        if not _is_onnx_gpu_runtime_error(exc):
            raise

        if _is_onnx_device_lost_error(exc):
            print(
                f"[recovery] ONNX GPU device-lost error for '{resolved_model_name}' "
                f"({exc}). Switching all ONNX models to CPU-only mode."
            )
            _set_cpu_only_onnx_mode()
            _reset_onnx_sessions()
        elif _is_onnx_out_of_memory_error(exc):
            print(
                f"[recovery] ONNX GPU OOM for '{resolved_model_name}' ({exc}). "
                "Falling back this model to CPU-only."
            )
            _onnx_cpu_only_models.add(resolved_model_name)
            _reset_onnx_session(resolved_model_name)
        else:
            print(
                f"[recovery] ONNX GPU runtime error for '{resolved_model_name}' ({exc}). "
                "Falling back this model to CPU-only."
            )
            _onnx_cpu_only_models.add(resolved_model_name)
            _reset_onnx_session(resolved_model_name)

        try:
            return _infer_once()
        except Exception as cpu_exc:
            raise RuntimeError(
                "ONNX inference failed on GPU and retry on CPU also failed"
            ) from cpu_exc


def get_session(model_name: str):
    if model_name not in _sessions:
        print(f"Loading rembg session: {model_name}")
        available = ort.get_available_providers()
        preferred_providers = _get_preferred_providers(model_name)
        # rembg forwards **kwargs to onnxruntime.InferenceSession
        _sessions[model_name] = new_session(model_name, providers=preferred_providers)
        print(
            f"Model ready: {model_name} | "
            f"providers={preferred_providers} | available={available}"
        )
    return _sessions[model_name]


def _get_preferred_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                test_tensor = torch.randn(1, device="cuda")
                del test_tensor
                torch.cuda.empty_cache()
                return "cuda"
            except Exception as exc:
                print(f"CUDA test failed: {exc}")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                test_tensor = torch.randn(1, device="mps")
                del test_tensor
                return "mps"
            except Exception as exc:
                print(f"MPS test failed: {exc}")
    except ImportError:
        pass

    return "cpu"


def _get_ben2():
    global _ben2
    if _ben2 is None:
        device = _get_preferred_torch_device()
        print(f"Loading BEN2 model on {device}…")
        from ben2 import BEN_Base  # type: ignore[import]

        model = BEN_Base.from_pretrained("PramaLLC/BEN2")
        _ben2 = model.to(device).eval()
        print(f"BEN2 ready ({device}).")
    return _ben2


def _get_rmbg2():
    global _rmbg2
    if _rmbg2 is None:
        device = _get_preferred_torch_device()
        print(f"Loading BRIA RMBG-2.0 model on {device}…")
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0",
                trust_remote_code=True,
            ).eval().to(device)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load BRIA RMBG-2.0. "
                "Request access on Hugging Face for 'briaai/RMBG-2.0' and authenticate "
                "locally (for example by setting HF_TOKEN)."
            ) from exc

        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ],
        )

        _rmbg2 = {
            "device": device,
            "model": model,
            "transform": transform_image,
            "torch": torch,
        }
        print(f"BRIA RMBG-2.0 ready ({device}).")

    return _rmbg2


def _run_rmbg2(image: Image.Image) -> Image.Image:
    runtime = _get_rmbg2()
    model = runtime["model"]
    device = runtime["device"]
    transform_image = runtime["transform"]
    torch = runtime["torch"]
    from torchvision.transforms.functional import to_pil_image

    input_images = transform_image(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    mask = to_pil_image(pred).resize(image.size, Image.LANCZOS)
    cutout = image.convert("RGBA")
    cutout.putalpha(mask)
    return cutout


def _to_luma_mask(image: Image.Image) -> Image.Image:
    return image if image.mode == "L" else image.convert("L")


def _normalize_model_name(model_name: str | None) -> str:
    return (model_name or DEFAULT_MODEL).strip().lower()


def _resolve_rembg_model_name(model_name: str) -> str:
    return REMBG_MODEL_ALIASES.get(model_name, model_name)


def _clear_torch_vram() -> None:
    """Aggressively clear torch VRAM to prevent OOM errors after inference."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:
        print(f"[memory] torch.cuda cleanup warning: {exc}")


# Maximum long edge passed to rembg per resolved model name.
# Models that operate at a fixed internal resolution (e.g. 1024×1024) gain nothing
# from a larger input but do cost extra VRAM for intermediate tensors.
# Heavy models that crash with DML OOM are capped at their native resolution.
_MODEL_MAX_INFERENCE_DIM: dict[str, int] = {
    # BRIA RMBG-1.4 runs at 1024×1024 internally.
    "bria-rmbg": 1536,
    # BiRefNet variants run at 1024×1024 internally; feed it exactly that to avoid DML OOM.
    "birefnet-general": 1024,
    "birefnet-general-lite": 1024,
    "birefnet-portrait": 1024,
    "birefnet-dis": 1024,
    "birefnet-hrsod": 1024,
    "birefnet-cod": 1024,
    "birefnet-massive": 1024,
    # u2net variants run at 320×320; 1024 is generous and still fine.
    "u2net": 1024,
    "u2netp": 768,
    "u2net_human_seg": 1024,
    "u2net_cloth_seg": 1024,
    # silueta runs at 320×320.
    "silueta": 768,
}
DEFAULT_MAX_INFERENCE_DIM = 1792


def _get_max_inference_dim(resolved_model_name: str | None = None) -> int:
    if resolved_model_name and resolved_model_name in _MODEL_MAX_INFERENCE_DIM:
        return _MODEL_MAX_INFERENCE_DIM[resolved_model_name]
    return DEFAULT_MAX_INFERENCE_DIM


def _resize_for_inference(
    image: Image.Image,
    resolved_model_name: str | None = None,
) -> Image.Image:
    """Downscale to the per-model max dimension on the long edge when the source is larger.
    
    Currently disabled by default (ENABLE_IMAGE_RESIZING=false).
    To enable, set ENABLE_IMAGE_RESIZING=true in environment.
    """
    if not ENABLE_IMAGE_RESIZING:
        return image

    max_dim_cap = _get_max_inference_dim(resolved_model_name)
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= max_dim_cap:
        return image
    scale = max_dim_cap / max_dim
    new_w, new_h = round(w * scale), round(h * scale)
    print(f"[resize] {w}×{h} → {new_w}×{new_h} (cap={max_dim_cap} for {resolved_model_name or 'default'})")
    return image.resize((new_w, new_h), Image.LANCZOS)


def _harden_alpha(output_image: Image.Image, hardness: float) -> Image.Image:
    """Push soft alpha values toward fully opaque or transparent.

    hardness=0   → no change (pure soft mask)
    hardness=1   → hard binary threshold at 128
    Values in between give a progressively tighter contrast curve.
    """
    hardness = max(0.0, min(1.0, float(hardness)))
    if hardness <= 0 or "A" not in output_image.getbands():
        return output_image

    spread = int((1.0 - hardness) * 127)
    low = 127 - spread   # alpha values at or below this become 0
    high = 128 + spread  # alpha values at or above this become 255

    def remap(p: int) -> int:
        if p <= low:
            return 0
        if p >= high:
            return 255
        return int((p - low) / (high - low) * 255)

    alpha = _to_luma_mask(output_image.getchannel("A"))
    hardened = alpha.point(remap)
    result = output_image.copy()
    result.putalpha(hardened)
    return result


def _apply_edge_feather(output_image: Image.Image, feather_px: float) -> Image.Image:
    radius = max(0.0, min(12.0, float(feather_px)))
    if radius <= 0 or "A" not in output_image.getbands():
        return output_image

    alpha = _to_luma_mask(output_image.getchannel("A"))
    softened_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=radius))
    softened = output_image.copy()
    softened.putalpha(softened_alpha)
    return softened

def _expand_mask(output_image: Image.Image, px: int = 1) -> Image.Image:
    if px <= 0 or "A" not in output_image.getbands():
        return output_image

    alpha = output_image.getchannel("A")
    expanded = alpha.filter(ImageFilter.MaxFilter(size=px * 2 + 1))

    result = output_image.copy()
    result.putalpha(expanded)
    return result

# Pre-load ONNX-backed rembg defaults at startup so first request is fast.
# BEN2 and RMBG-2.0 are torch-backed and remain lazy-loaded.
_default_rembg_model_name = REMBG_MODEL_ALIASES.get(DEFAULT_MODEL)
if _default_rembg_model_name is not None:
    get_session(_default_rembg_model_name)

# ── Inference semaphore ───────────────────────────────────────────────────────
# Limit concurrent inference to 1.  Downloads happen outside the lock so
# multiple requests can fetch their source images in parallel while only
# one occupies VRAM at a time, preventing OOM on DML/CUDA devices.
_inference_semaphore = asyncio.Semaphore(1)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="BRIA remove-bg service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class RemoveBgRequest(BaseModel):
    imageUrl: str
    model: str | None = None
    # Alpha matting refines soft/blurry edges (e.g. out-of-focus limbs).
    # Requires `pip install pymatting` (listed in requirements.txt).
    alphaMattingEnabled: bool = DEFAULT_ALPHA_MATTING_ENABLED
    alphaMattingFgThreshold: int = DEFAULT_ALPHA_MATTING_FG_THRESHOLD  # 0-255: above → definite foreground
    alphaMattingBgThreshold: int = DEFAULT_ALPHA_MATTING_BG_THRESHOLD  # 0-255: below → definite background
    alphaMattingErodeSize: int = DEFAULT_ALPHA_MATTING_ERODE_SIZE      # px to shrink unknown region before matting
    edgeFeatherPx: float = DEFAULT_EDGE_FEATHER_PX
    maskHardness: float = DEFAULT_MASK_HARDNESS  # 0 = soft as-is, 1 = hard binary threshold at 128


@app.post("/remove-bg")
async def remove_background(body: RemoveBgRequest) -> Response:
    t0 = time.perf_counter()
    url = body.imageUrl.strip()
    if not url:
        raise HTTPException(status_code=400, detail="imageUrl is required")

    model_name = _normalize_model_name(body.model)
    if model_name not in SUPPORTED_REMOVE_BG_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{model_name}'. "
                f"Allowed: {list(SUPPORTED_REMOVE_BG_MODELS)}"
            ),
        )

    # Download the source image
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            image_bytes = resp.content
        t_download = time.perf_counter()
        print(f"[timing] download: {t_download - t0:.2f}s  ({len(image_bytes)//1024} KB)")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download image ({exc.response.status_code}): {url}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download image: {exc}",
        ) from exc

    # Remove background — serialised through _inference_semaphore so only one
    # request occupies VRAM at a time while downloads can still overlap.
    input_image: Image.Image | None = None
    infer_image: Image.Image | None = None
    output_image: Image.Image | None = None
    resolved_for_resize = _resolve_rembg_model_name(model_name) if model_name not in (BEN2_MODEL, BRIA_20_MODEL) else None
    t_wait_start = time.perf_counter()
    async with _inference_semaphore:
        t_wait = time.perf_counter() - t_wait_start
        if t_wait > 0.1:
            print(f"[timing] queue wait: {t_wait:.2f}s")
        try:
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            image_bytes = b""  # release raw bytes immediately
            infer_image = _resize_for_inference(input_image, resolved_for_resize)
            if infer_image is not input_image:
                input_image.close()
                input_image = None
            t_decode = time.perf_counter()
            size_info = f"{infer_image.size[0]}\u00d7{infer_image.size[1]}"
            print(f"[timing] decode:   {t_decode - t_download:.2f}s  ({size_info})")
            print(
                "[settings] cutout: "
                f"model={model_name} | "
                f"alphaMattingEnabled={body.alphaMattingEnabled} | "
                f"alphaMattingFgThreshold={body.alphaMattingFgThreshold} | "
                f"alphaMattingBgThreshold={body.alphaMattingBgThreshold} | "
                f"alphaMattingErodeSize={body.alphaMattingErodeSize} | "
                f"edgeFeatherPx={body.edgeFeatherPx} | "
                f"maskHardness={body.maskHardness}"
            )

            if model_name == BEN2_MODEL:
                ben2_model = _get_ben2()
                output_image = ben2_model.inference(
                    infer_image.convert("RGB"),
                    refine_foreground=True,
                )
                if isinstance(output_image, list):
                    output_image = output_image[0]
                if output_image.mode != "RGBA":
                    output_image = output_image.convert("RGBA")
            elif model_name == BRIA_20_MODEL:
                output_image = _run_rmbg2(infer_image)
            else:
                output_image = _run_rembg_inference_with_recovery(
                    infer_image=infer_image,
                    model_name=model_name,
                    alpha_matting_enabled=body.alphaMattingEnabled,
                    alpha_matting_fg_threshold=body.alphaMattingFgThreshold,
                    alpha_matting_bg_threshold=body.alphaMattingBgThreshold,
                    alpha_matting_erode_size=body.alphaMattingErodeSize,
                )
            t_infer = time.perf_counter()
            print(f"[timing] inference: {t_infer - t_decode:.2f}s")

            # Clear torch VRAM immediately after inference to prevent OOM from gradual accumulation
            _clear_torch_vram()
            gc.collect()

            infer_image.close()
            infer_image = None
            if input_image is not None:
                input_image.close()
                input_image = None

            # Harden before feathering so the blur operates on already-clean edges.
            output_image = _harden_alpha(output_image, body.maskHardness)
            if model_name == "u2-net":
                output_image = _expand_mask(output_image, 1) # u2 net fix

            output_image = _apply_edge_feather(output_image, body.edgeFeatherPx)
            t_post = time.perf_counter()
            print(f"[timing] post-proc: {t_post - t_infer:.2f}s")
        except Exception as exc:
            for img in (input_image, infer_image, output_image):
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
            gc.collect()
            raise HTTPException(
                status_code=500,
                detail=f"Background removal failed: {exc}",
            ) from exc

    # Encode as PNG (outside the semaphore — purely CPU/memory work)
    out_bytes = io.BytesIO()
    output_image.save(out_bytes, format="PNG")
    output_image.close()
    gc.collect()
    out_bytes.seek(0)
    t_encode = time.perf_counter()
    print(f"[timing] encode:    {t_encode - t_post:.2f}s | TOTAL: {t_encode - t0:.2f}s")

    return Response(content=out_bytes.read(), media_type="image/png")


# ── LaMa inpainting (background fill) ────────────────────────────────────────
# Lazy-loaded on first use so the service starts fast even if LaMa isn't needed.
_lama_by_device: dict[str, object] = {}


def _get_preferred_lama_device() -> str:
    """Get the preferred device for LaMa inpainting: cuda if available, else mps (Apple Silicon), else cpu."""
    if LAMA_FORCE_CPU:
        return "cpu"
    return _get_preferred_torch_device()


def _get_lama(device: str | None = None):
    resolved_device = device or _get_preferred_lama_device()
    if resolved_device not in _lama_by_device:
        print(f"Loading LaMa inpainting model on {resolved_device}…")
        from simple_lama_inpainting import SimpleLama  # type: ignore[import]

        _lama_by_device[resolved_device] = SimpleLama(device=resolved_device)
        print(f"LaMa ready ({resolved_device}).")
    return _lama_by_device[resolved_device]


def _downscale_for_lama(
    source_rgb: Image.Image,
    mask: Image.Image,
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    """Shrink RGB + mask together so LaMa does not run at full photo resolution."""
    w, h = source_rgb.size
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.NEAREST)
    max_edge = max(w, h)
    cap = max(256, LAMA_MAX_LONG_EDGE)
    if max_edge <= cap:
        return source_rgb, mask, (w, h)
    scale = cap / max_edge
    nw, nh = round(w * scale), round(h * scale)
    print(
        f"[inpaint] LaMa input downscale: {w}×{h} → {nw}×{nh} "
        f"(LAMA_MAX_LONG_EDGE={cap})"
    )
    small_rgb = source_rgb.resize((nw, nh), Image.LANCZOS)
    small_mask = mask.resize((nw, nh), Image.NEAREST)
    return small_rgb, small_mask, (w, h)


def _maybe_upscale_lama_result(
    result: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    if result.size == target_size:
        return result
    print(f"[inpaint] LaMa output upscale: {result.size[0]}×{result.size[1]} → {target_size[0]}×{target_size[1]}")
    return result.resize(target_size, Image.LANCZOS)


async def _run_lama_fill_with_recovery(source_rgb: Image.Image, mask: Image.Image) -> Image.Image:
    lama_rgb, lama_mask, out_size = _downscale_for_lama(source_rgb, mask)
    preferred_device = _get_preferred_lama_device()
    lama = _get_lama(preferred_device)

    def _infer() -> Image.Image:
        print(
            f"[inpaint] LaMa inference start ({preferred_device}) "
            f"{lama_rgb.size[0]}×{lama_rgb.size[1]}"
        )
        t_infer = time.perf_counter()
        out = lama(lama_rgb, lama_mask)
        print(f"[inpaint] LaMa inference done in {time.perf_counter() - t_infer:.2f}s")
        return out

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_infer),
            timeout=LAMA_INFERENCE_TIMEOUT_SEC,
        )
    except Exception as exc:
        if preferred_device == "cpu":
            raise RuntimeError(f"LaMa inference failed on CPU: {exc}") from exc

        print(
            f"[recovery] LaMa inference failed on {preferred_device} ({exc}). "
            "Falling back to CPU for this request."
        )
        lama_cpu = _get_lama("cpu")

        def _infer_cpu() -> Image.Image:
            print(
                f"[inpaint] LaMa inference start (cpu fallback) "
                f"{lama_rgb.size[0]}×{lama_rgb.size[1]}"
            )
            t_infer = time.perf_counter()
            out = lama_cpu(lama_rgb, lama_mask)
            print(f"[inpaint] LaMa inference done in {time.perf_counter() - t_infer:.2f}s")
            return out

        result = await asyncio.wait_for(
            asyncio.to_thread(_infer_cpu),
            timeout=LAMA_INFERENCE_TIMEOUT_SEC,
        )

    return _maybe_upscale_lama_result(result, out_size)


class InpaintRequest(BaseModel):
    imageUrl: str
    model: str | None = None  # rembg model used to produce the mask
    returnCutout: bool = False  # NEW: also return the subject cutout
    # Alpha matting refines soft/blurry edges (e.g. out-of-focus limbs).
    # Requires `pip install pymatting` (listed in requirements.txt).
    alphaMattingEnabled: bool = DEFAULT_ALPHA_MATTING_ENABLED
    alphaMattingFgThreshold: int = DEFAULT_ALPHA_MATTING_FG_THRESHOLD  # 0-255: above → definite foreground
    alphaMattingBgThreshold: int = DEFAULT_ALPHA_MATTING_BG_THRESHOLD  # 0-255: below → definite background
    alphaMattingErodeSize: int = DEFAULT_ALPHA_MATTING_ERODE_SIZE      # px to shrink unknown region before matting


@app.post("/inpaint")
async def inpaint_background(body: InpaintRequest) -> Response:
    """Remove the foreground subject and fill the gap with LaMa inpainting.
    
    Optionally returns both inpainted background and subject cutout.
    """
    t0 = time.perf_counter()
    url = body.imageUrl.strip()
    if not url:
        raise HTTPException(status_code=400, detail="imageUrl is required")

    model_name = _normalize_model_name(body.model)
    if model_name not in SUPPORTED_REMOVE_BG_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{model_name}'. "
                f"Allowed: {list(SUPPORTED_REMOVE_BG_MODELS)}"
            ),
        )

    # Download source image
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            image_bytes = resp.content
        t_download = time.perf_counter()
        print(f"[inpaint timing] download: {t_download - t0:.2f}s  ({len(image_bytes) // 1024} KB)")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download image ({exc.response.status_code}): {url}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download image: {exc}") from exc

    source_image: Image.Image | None = None
    infer_image: Image.Image | None = None
    cutout: Image.Image | None = None
    result: Image.Image | None = None
    resolved_for_resize = _resolve_rembg_model_name(model_name) if model_name not in (BEN2_MODEL, BRIA_20_MODEL) else None
    t_wait_start = time.perf_counter()
    async with _inference_semaphore:
        t_wait = time.perf_counter() - t_wait_start
        if t_wait > 0.1:
            print(f"[inpaint timing] queue wait: {t_wait:.2f}s")
        try:
            source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_bytes = b""

            # Resize for inference
            infer_rgb = _resize_for_inference(source_image.convert("RGBA"), resolved_for_resize)
            infer_image = infer_rgb
            t_decode = time.perf_counter()
            print(f"[inpaint timing] decode: {t_decode - t_download:.2f}s  ({infer_image.size[0]}×{infer_image.size[1]})")

            print(
                "[inpaint settings] cutout: "
                f"alphaMattingEnabled={body.alphaMattingEnabled} | "
                f"alphaMattingFgThreshold={body.alphaMattingFgThreshold} | "
                f"alphaMattingBgThreshold={body.alphaMattingBgThreshold} | "
                f"alphaMattingErodeSize={body.alphaMattingErodeSize}"
            )

            # Step 1: get subject mask via selected cutout model
            if model_name == BEN2_MODEL:
                ben2_model = _get_ben2()
                cutout = ben2_model.inference(
                    infer_image.convert("RGB"),
                    refine_foreground=True,
                )
                if isinstance(cutout, list):
                    cutout = cutout[0]
                if cutout.mode != "RGBA":
                    cutout = cutout.convert("RGBA")
            elif model_name == BRIA_20_MODEL:
                cutout = _run_rmbg2(infer_image)
            else:
                cutout = _run_rembg_inference_with_recovery(
                    infer_image=infer_image,
                    model_name=model_name,
                    alpha_matting_enabled=body.alphaMattingEnabled,
                    alpha_matting_fg_threshold=body.alphaMattingFgThreshold,
                    alpha_matting_bg_threshold=body.alphaMattingBgThreshold,
                    alpha_matting_erode_size=body.alphaMattingErodeSize,
                )
            t_mask = time.perf_counter()
            print(f"[inpaint timing] mask:   {t_mask - t_decode:.2f}s")

            # Clear torch VRAM after cutout generation to prevent OOM
            _clear_torch_vram()
            gc.collect()

            # If requested, return both results
            if body.returnCutout:
                alpha = cutout.split()[3]  # L image, subject = 255, bg = 0
                mask = alpha.filter(ImageFilter.MaxFilter(7))
                source_rgb = infer_image.convert("RGB")
                result = await _run_lama_fill_with_recovery(source_rgb, mask)
                t_fill = time.perf_counter()
                print(f"[inpaint timing] lama:   {t_fill - t_mask:.2f}s")
                _clear_torch_vram()
                gc.collect()
                result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
            else:
                alpha = cutout.split()[3]  # L image, subject = 255, bg = 0
                mask = alpha.filter(ImageFilter.MaxFilter(7))
                source_rgb = infer_image.convert("RGB")
                result = await _run_lama_fill_with_recovery(source_rgb, mask)
                t_fill = time.perf_counter()
                print(f"[inpaint timing] lama:   {t_fill - t_mask:.2f}s")
                _clear_torch_vram()
                gc.collect()
                result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))

        except Exception as exc:
            for img in (source_image, infer_image, cutout, result):
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
            gc.collect()
            raise HTTPException(
                status_code=500,
                detail=f"Inpainting failed: {exc}",
            ) from exc
        finally:
            for img in (source_image, infer_image):
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass

    # Encode result (outside the semaphore — purely CPU/memory work)
    if body.returnCutout:
        import base64
        from io import BytesIO
        bg_buffer = BytesIO()
        result.save(bg_buffer, format="JPEG", quality=92)
        bg_b64 = base64.b64encode(bg_buffer.getvalue()).decode()
        cutout_buffer = BytesIO()
        cutout.save(cutout_buffer, format="PNG")
        cutout_b64 = base64.b64encode(cutout_buffer.getvalue()).decode()
        result.close()
        cutout.close()
        gc.collect()
        return JSONResponse(content={
            "inpaintedBackground": f"data:image/jpeg;base64,{bg_b64}",
            "subjectCutout": f"data:image/png;base64,{cutout_b64}",
        })

    out_bytes = io.BytesIO()
    result.save(out_bytes, format="JPEG", quality=92)
    result.close()
    cutout.close()
    gc.collect()
    out_bytes.seek(0)
    t_encode = time.perf_counter()
    print(f"[inpaint timing] encode: {t_encode - t_fill:.2f}s | TOTAL: {t_encode - t0:.2f}s")

    return Response(content=out_bytes.read(), media_type="image/jpeg")




@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "supported_models": list(SUPPORTED_REMOVE_BG_MODELS),
        "rembg_aliases": REMBG_MODEL_ALIASES,
        "onnx_provider_override": _onnx_providers_override,
        "onnx_cpu_only_models": sorted(_onnx_cpu_only_models),
        "loaded_models": list(_sessions.keys()),
        "ben2_loaded": _ben2 is not None,
        "rmbg2_loaded": _rmbg2 is not None,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
