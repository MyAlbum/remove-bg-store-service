from pydantic import BaseModel


class RemoveBgStoreRequest(BaseModel):
    imageUrl: str
    model: str | None = None
    alphaMattingFgThreshold: float | None = None
    alphaMattingBgThreshold: float | None = None
    alphaMattingErodeSize: float | None = None
    edgeFeatherPx: float | None = None

    # Compatibility aliases used by the current Next.js route.
    NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_FG_THRESHOLD: float | None = None
    NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_BG_THRESHOLD: float | None = None
    NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_ERODE_SIZE: float | None = None
    NEXT_PUBLIC_REMOVE_BG_EDGE_FEATHER_PX: float | None = None
