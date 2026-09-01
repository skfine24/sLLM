"""DeepSeek-V4 image preprocessing (numpy; Pillow only bytes->PIL decode).

Port of `ref/hf_sources/dsv4/image_processor.py` (OpenAI-style message blobs
in, sentinel token blocks + patch tensors out) with numpy in place of torch.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


@dataclass
class ImageInput:
    start: int
    patches: np.ndarray          # (n_patch, 3, patch, patch) float32
    n_vit_h: int
    n_vit_w: int
    types: np.ndarray            # (block,) int64 sentinel types
    perm: np.ndarray             # (n_aligned,) int64 alignment-reorder


@dataclass
class VisionArgs:
    patch_size: int = 14
    downsample_ratio: int = 3
    max_n_token: int = 384
    min_pixels: int = 147456
    max_wh_ratio: float | None = 8.0


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch_size * downsample_ratio / width,
                   max_h * patch_size * downsample_ratio / height)
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width,
                                               patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size,
                downsample_ratio, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width,
                                               patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = \
            solve_resize_ratio(height, width, patch_size, downsample_ratio,
                               budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def load_image_bytes(record) -> bytes:
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})
    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        with open(url, "rb") as file:
            return file.read()
    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def load_image(record, args: VisionArgs | None = None):
    args = args or VisionArgs()
    p = args.patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    width, height = image.size
    if args.max_wh_ratio is not None and width > height * args.max_wh_ratio:
        width = height * args.max_wh_ratio
    if 0 < width * height < args.min_pixels:
        ratio = (args.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, args.downsample_ratio,
        args.max_n_token)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if args.max_wh_ratio is not None and \
            image.width >= args.max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height),
                             color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    x = (x - 0.5) / 0.5  # float32 (bf16 cast is a cluster-path detail)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).transpose(1, 3, 0, 2, 4)
    patches = patches.reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h \
        + [IMAGE_PAD] * (row_len * pad_h)
    order = np.arange(rows * row_len).reshape(rows // 2, 2, row_len)
    order = order.swapaxes(1, 2).reshape(-1)
    image_idx = np.full((rows * row_len,), -1, dtype=np.int64)
    image_idx.reshape(rows, row_len)[:n_llm_h, :n_llm_w] = np.arange(
        n_llm_h * n_llm_w).reshape(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = ([IMAGE_PAD] * compress_pad + [IMAGE_START]
             + np.asarray(types, np.int64)[order].tolist()
             + [IMAGE_PAD] * pad_last + [IMAGE_END])
    return np.asarray(types, np.int64), perm.astype(np.int64)


def expand_image_placeholders(prompt_ids, images, args: VisionArgs | None = None,
                              vocab_size: int = 129280,
                              image_placeholder_id: int | None = None):
    """Expand `<image>` placeholder tokens into sentinel blocks.

    Returns (llm_ids, list[ImageInput] | None). Every image block is placed
    at the placeholder's absolute position and its sentinel ids are
    `vocab_size + type` (the encoding contract for embedding lookup).
    """
    args = args or VisionArgs()
    tokens, image_inputs = [], []
    it = iter(images)
    ph = image_placeholder_id
    for tok in prompt_ids:
        if ph is not None and tok == ph:
            patches, nvh, nw, nlh, nlw = load_image(next(it), args)
            types, perm = build_image_block(nlh, nlw, len(tokens))
            image_inputs.append(ImageInput(len(tokens), patches, nvh, nw,
                                           types, perm))
            tokens += (vocab_size + types).tolist()
        else:
            tokens.append(tok)
    if not image_inputs:
        return tokens, None
    return tokens, image_inputs
