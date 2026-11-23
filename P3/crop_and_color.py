"""Khang Lam | ECE 539 | LEGO crop preparation and fixed color matching.

"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 224

# Selected opaque colors from https://cdn.rebrickable.com/media/downloads/colors.csv.gz
# Keys are Rebrickable color IDs. The numbers between keys are intentionally unused.
DEFAULT_PALETTE = {
    0: ("Black", "05131D"),
    1: ("Blue", "0055BF"),
    2: ("Green", "237841"),
    3: ("Dark Turquoise", "008F9B"),
    4: ("Red", "C91A09"),
    10: ("Bright Green", "4B9F4A"),
    14: ("Yellow", "F2CD37"),
    15: ("White", "FFFFFF"),
    19: ("Tan", "E4CD9E"),
    25: ("Orange", "FE8A18"),
    26: ("Magenta", "923978"),
    27: ("Lime", "BBE90B"),
    28: ("Dark Tan", "958A73"),
    29: ("Bright Pink", "E4ADC8"),
    70: ("Reddish Brown", "582A12"),
    71: ("Light Bluish Gray", "A0A5A9"),
    72: ("Dark Bluish Gray", "6C6E68"),
    73: ("Medium Blue", "5A93DB"),
    84: ("Medium Nougat", "AA7D55"),
    85: ("Dark Purple", "3F3691"),
    272: ("Dark Blue", "0A3463"),
    288: ("Dark Green", "184632"),
    320: ("Dark Red", "720E0F"),
    322: ("Medium Azure", "36AEBF"),
}


@dataclass(frozen=True)
class ColorMatch:
    color_id: int
    color_name: str
    median_rgb: tuple[int, int, int]
    delta_e: float  # distance to the selected swatch, in CIELAB units


def crop_bricks(image: Image.Image, detections) -> list[Image.Image]:
    """Return one 224 x 224 RGB crop per detector row, in the original order.

    Supply a PIL image in the orientation used during detection and an N x 4
    or N x 6 array of (x1, y1, x2, y2[, confidence, class]) in original pixels.
    Clipping allows partly visible bricks. An unusable box raises an error
    naming its zero-based row so shape, color and box lists stay aligned.
    """
    image = image.convert("RGB")
    if hasattr(detections, "detach"):  
        detections = detections.detach().cpu().numpy()
    rows = np.asarray(detections, dtype=np.float64)
    if rows.shape == (0,):
        return []
    if rows.ndim != 2 or rows.shape[1] not in (4, 6):
        raise ValueError("Expected detector rows in N x 4 or N x 6 pixel xyxy format.")

    crops = []
    for index, row in enumerate(rows):
        coords = row[:4]
        if not np.isfinite(coords).all():
            raise ValueError(f"Detection {index} contains a non-finite coordinate.")
        left, top, right, bottom = np.clip(coords, 0, [*image.size, *image.size])
        if right <= left or bottom <= top:
            raise ValueError(f"Detection {index} has no area inside the image.")
        
        bounds = (int(np.floor(left)), int(np.floor(top)),
                  int(np.ceil(right)), int(np.ceil(bottom)))
        crop = image.crop(bounds).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        crops.append(crop)
    return crops


def load_palette(csv_path: str | Path) -> dict[int, tuple[str, str]]:
    """Read a Rebrickable-format CSV with id, name, rgb and optional is_trans.

    Choose rows for the opaque colors in your collection. A full catalog also
    contains specialty finishes with overlapping RGB values. The matcher can
    only distinguish colors by their RGB swatches.
    """
    palette = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"id", "name", "rgb"}.issubset(reader.fieldnames or []):
            raise ValueError("Palette CSV needs id, name and rgb columns.")
        for row in reader:
            color_id = int(row["id"])
            if color_id in (-1, 9999) or row.get("is_trans", "").strip().lower() in {"t", "true", "1"}:
                continue
            if color_id in palette:
                raise ValueError(f"Duplicate color ID {color_id}.")
            palette[color_id] = (row["name"], row["rgb"].strip().lstrip("#"))
    if not palette:
        raise ValueError("Palette has no candidate colors.")
    return palette


def _lab(rgb):
    """Convert sRGB values on 0..255 to CIELAB under the D65 white point."""
    s = np.asarray(rgb, dtype=np.float64) / 255
    linear = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    rgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
    xyz = (linear @ rgb_to_xyz.T) / [0.95047, 1.0, 1.08883]
    delta = 6 / 29
    fx, fy, fz = np.moveaxis(
        np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29), -1, 0
    )
    return np.stack((116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)), axis=-1)


def match_colors(crops, palette=None) -> list[ColorMatch]:
    """Match RGB PIL crops to the nearest reference using CIE76 distance.

    Each channel median uses the central half of the crop's width and height.
    This reduces the influence of the background at the four edges. Exact ties
    go to the smaller Rebrickable ID. Requires no fitted parameters.
    """
    if palette is None:
        palette = DEFAULT_PALETTE
    if not palette:
        raise ValueError("Palette has no candidate colors.")

    ids = sorted(palette)
    swatches = [bytes.fromhex(palette[color_id][1]) for color_id in ids]
    if any(len(swatch) != 3 for swatch in swatches):
        raise ValueError("Expected a six-digit RGB hex value for each palette color.")
    references = _lab([list(swatch) for swatch in swatches])

    matches = []
    for crop in crops:
        w, h = crop.size
        box = (w // 4, h // 4, w - w // 4, h - h // 4)
        center = np.asarray(crop.convert("RGB").crop(box))
        median = np.median(center, axis=(0, 1))
        distances = np.linalg.norm(references - _lab(median), axis=1)
        best = int(distances.argmin())
        color_id = ids[best]
        matches.append(ColorMatch(color_id, palette[color_id][0],
                                  tuple(int(x) for x in np.rint(median)), float(distances[best])))
    return matches
