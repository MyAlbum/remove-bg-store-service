from collections import deque
from dataclasses import dataclass
from io import BytesIO

from PIL import Image


ALPHA_THRESHOLD = 128
MIN_COMPONENT_PIXELS = 50
CONNECTIVITY_8 = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]


@dataclass
class SubjectMetrics:
    width: int
    height: int
    foreground_ratio: float


def _alpha_binary(png_bytes: bytes) -> tuple[list[int], int, int]:
    image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    width, height = image.size
    alpha = image.getchannel("A")
    alpha_data = list(alpha.getdata())
    binary = [1 if a > ALPHA_THRESHOLD else 0 for a in alpha_data]
    return binary, width, height


def compute_subject_metrics(png_bytes: bytes) -> SubjectMetrics:
    binary, width, height = _alpha_binary(png_bytes)
    total = width * height
    foreground = sum(binary)
    foreground_ratio = (foreground / total) if total > 0 else 0.0
    return SubjectMetrics(width=width, height=height, foreground_ratio=foreground_ratio)


def compute_subject_objects(png_bytes: bytes) -> list[dict]:
    binary, width, height = _alpha_binary(png_bytes)
    visited = [0] * (width * height)
    objects: list[dict] = []

    for i in range(width * height):
        if binary[i] == 0 or visited[i]:
            continue

        visited[i] = 1
        q = deque([i])
        size = 0
        min_x = width
        min_y = height
        max_x = -1
        max_y = -1

        while q:
            cur = q.pop()
            size += 1
            x = cur % width
            y = cur // width

            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

            for dx, dy in CONNECTIVITY_8:
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                ni = ny * width + nx
                if binary[ni] == 0 or visited[ni]:
                    continue
                visited[ni] = 1
                q.append(ni)

        if size >= MIN_COMPONENT_PIXELS and max_x >= min_x and max_y >= min_y:
            objects.append(
                {
                    "x": min_x,
                    "y": min_y,
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                }
            )

    return objects
