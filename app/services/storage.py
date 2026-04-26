import hashlib
from urllib.parse import urlparse


def normalize_image_url(raw_url: str) -> str:
    try:
        parsed = urlparse(raw_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return raw_url
    except Exception:
        return raw_url


def build_filename(input_data: dict) -> str:
    value = "|".join(
        [
            "v1",
            normalize_image_url(str(input_data["imageUrl"])),
            str(input_data["model"]),
            str(input_data["alphaMattingFgThreshold"]),
            str(input_data["alphaMattingBgThreshold"]),
            str(input_data["alphaMattingErodeSize"]),
            str(input_data["edgeFeatherPx"]),
        ]
    )
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"cutout-{digest}.png"
