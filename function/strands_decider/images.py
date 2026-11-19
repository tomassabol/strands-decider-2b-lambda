"""Images for HiddenStateEngine: decoding, sizing, and where they go in the prompt.

They are prepared as the Qwen image processor prepares them, so the vision tower sees the
pixels the model was built for. HiddenStateEngine imports this module only for a request with images.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import re

from PIL import Image, ImageOps

from .prompting import render_state
from .schema import Content

VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"


def render_image_state(state: Content, n_images: int) -> str:
    """`render_state(state)` with one unexpanded placeholder per image, before the text.

    Images come first so the expensive part of the prefix (the vision tower and the
    image tokens through the decoder) is identical across the questions of a request.
    """
    text = render_state(state)
    placeholders = "\n".join(f"{VISION_START}{IMAGE_PAD}{VISION_END}" for _ in range(n_images))
    body = text[len("<state>\n") : -len("\n</state>\n")]
    inner = f"{placeholders}\n{body}" if body else placeholders
    return f"<state>\n{inner}\n</state>\n"


# The formats the API accepts. Anything else Pillow can open (EPS through Ghostscript, for
# one) is refused before it is parsed further.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")


def decode_image(data: str, max_pixels: int) -> Image.Image:
    """A base64 image (optionally a `data:image/...;base64,` URI) as an upright RGB image.

    The size is checked from the header before any pixel is decoded, so a small file that
    expands to a huge bitmap is refused cheaply.
    """
    payload = re.sub(r"^data:image/[\w.+-]+;base64,", "", data.strip())
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image is not valid base64: {e}") from e
    try:
        img = Image.open(io.BytesIO(raw), formats=IMAGE_FORMATS)
    except Exception as e:
        raise ValueError(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from e
    w, h = img.size
    if w * h > max_pixels:
        raise ValueError(f"image is {w}x{h}; at most {max_pixels:,} pixels are accepted")
    try:
        img.load()
        # Phone photos store their rotation in EXIF. A malformed EXIF block raises
        # SyntaxError or struct.error here, which would otherwise surface as HTTP 500.
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ValueError(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)  # transparent areas read as white, not black
    return img.convert("RGB")


def fit_image(img: Image.Image, long_side: int) -> Image.Image:
    """RGB, scaled down (never up) so the longer side is at most `long_side`."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > long_side:
        s = long_side / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return img


# What Qwen2VLImageProcessorPil does to an image after fit_image, for a backend that is
# handed pixels instead of running the processor: patch 16 and merge 2 make a 32-pixel grid,
# and Qwen3.5-2B-Base's preprocessor_config.json bounds its area.
GRID_FACTOR = 32
GRID_MIN_PIXELS = 65536
GRID_MAX_PIXELS = 16777216


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """The processor's smart_resize: sides that are multiples of 32, the area in its bounds.
    An image more than 200 times as long as it is wide is refused, as the processor refuses it."""
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {ratio}")
    f = GRID_FACTOR
    h_bar, w_bar = round(height / f) * f, round(width / f) * f
    if h_bar * w_bar > GRID_MAX_PIXELS:
        beta = math.sqrt(height * width / GRID_MAX_PIXELS)
        h_bar = max(f, math.floor(height / beta / f) * f)
        w_bar = max(f, math.floor(width / beta / f) * f)
    elif h_bar * w_bar < GRID_MIN_PIXELS:
        beta = math.sqrt(GRID_MIN_PIXELS / (height * width))
        h_bar, w_bar = math.ceil(height * beta / f) * f, math.ceil(width * beta / f) * f
    return h_bar, w_bar


def to_grid(img: Image.Image) -> tuple[bytes, int, int, int]:
    """The image resized as the processor resizes it (bicubic, to its grid), as RGB bytes,
    with its width, its height and its tokens after the patch merger."""
    w, h = img.size
    gh, gw = smart_resize(h, w)
    if (gw, gh) != (w, h):
        img = img.resize((gw, gh), Image.BICUBIC)
    return img.tobytes(), gw, gh, (gh // GRID_FACTOR) * (gw // GRID_FACTOR)


def state_segments(state: Content, n_images: int) -> list[str]:
    """render_image_state's text around its images: n_images + 1 segments, with image i
    between segment i and segment i + 1. A state that holds an `<|image_pad|>` of its own is
    refused."""
    text = render_image_state(state, n_images)
    n_pads = text.count(IMAGE_PAD)
    if n_pads != n_images:
        raise ValueError(f"the state holds {n_pads - n_images} image placeholder(s) of its own")
    return text.split(f"{VISION_START}{IMAGE_PAD}{VISION_END}")
