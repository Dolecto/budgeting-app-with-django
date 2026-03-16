"""
stages_1_2.py
─────────────
Stage 1 – Image preprocessing
    1a. Contrast boost on a detection copy for better edge finding
    1b. Document boundary detection (GrabCut primary, Canny fallback)
    1c. Perspective correction (four-point transform → tight document crop)
        The skew angle of the document is derived directly from the contour
        corners at this step — NOT from Hough lines run on the cropped image.
    1d. Orientation correction (coarse via Tesseract OSD, fine from contour)
    1e. Output: RGB image of the document, upright, with padding on all sides

Stage 2 – Tiling
    2a. Compute tile height from *document width* × A4 aspect ratio
    2b. Slice with ≥15 % overlap
    2c. Pad each tile top/bottom so the document portion sits centred on a
        white A4-proportioned canvas → output resembles a photo of an A4
        document taken at a slight distance

Dependencies: opencv-python, Pillow, pytesseract, numpy
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

A4_ASPECT = 297 / 210           # height / width ≈ 1.414
PADDING_FRACTION = 0.05         # background padding around document (5 % of doc width)
OVERLAP_FRACTION = 0.15         # tile overlap (≥ 15 % as required)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Tile:
    """One A4-proportioned tile ready for UVDoc."""
    image: Image.Image      # full tile (document + white padding)
    index: int
    doc_y_start: int        # row in the *cropped document* where this tile starts
    doc_y_end: int          # row in the *cropped document* where this tile ends
    padded_top: int         # white rows added above the document slice
    padded_bottom: int      # white rows added below the document slice


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1a. Detection-only contrast boost ────────────────────────────────────────

def _detection_copy(image: np.ndarray) -> np.ndarray:
    """
    Return a contrast-boosted greyscale copy used *only* for edge/contour
    detection.  The original image is never modified here.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # CLAHE (adaptive histogram equalisation) works better than a global
    # contrast stretch on receipts with uneven lighting or shadows.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ── 1b. Document boundary detection ──────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[s.argmin()]       # top-left  (smallest x+y)
    rect[2] = pts[s.argmax()]       # bottom-right (largest x+y)
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[diff.argmin()]    # top-right  (smallest y-x)
    rect[3] = pts[diff.argmax()]    # bottom-left (largest y-x)
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-correct crop to the quadrilateral defined by *pts*."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl),
    ))
    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl),
    ))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


def _skew_from_corners(corners: np.ndarray) -> float:
    """
    Derive the residual skew angle (degrees) from the four ordered corners
    [top-left, top-right, bottom-right, bottom-left] of the document contour.

    We use the top edge (TL → TR) as the reference because:
      • It is the longest near-horizontal edge on a portrait receipt.
      • It is computed from the actual document boundary, not from interior
        text lines or crumple artifacts — making it far more reliable than
        running a Hough transform on the cropped image.

    Returns the angle to rotate counter-clockwise to level the document.
    A positive return value means the top-right corner is higher than the
    top-left (document tilted clockwise); negative means the reverse.
    """
    tl, tr = corners[0], corners[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    if dx == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _quad_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Given a binary foreground mask, find the largest quadrilateral contour
    and return its four corner points (float32, shape 4×2), or None.

    Tries progressively looser epsilon values so crumpled, non-straight
    edges still collapse to exactly 4 points.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = mask.shape[0] * mask.shape[1]

    for c in contours[:5]:
        if cv2.contourArea(c) < image_area * 0.10:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

    return None


def _detect_via_grabcut(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Use GrabCut to segment the document (light foreground) from the
    background (dark table/surface) and return a quadrilateral.

    GrabCut works on colour, not edge geometry, so it handles crumpled
    receipts where Canny only finds fragmented edges that never close into
    a clean polygon.

    Strategy
    ────────
    We initialise the foreground rectangle as a 10 % inset from the image
    border — this tells GrabCut that the outer strip is likely background
    and the centre region likely contains the document.  The receipt's white
    colour against the dark table makes this a strong, reliable signal.
    """
    h, w = image.shape[:2]
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask      = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # Pixels labelled probable or definite foreground
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)

    return _quad_from_mask(fg_mask)


def _detect_via_canny(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Fallback detector: Canny edge map → largest quadrilateral contour.
    Works well when there is a clear luminance boundary between document
    and background, but fails on crumpled receipts with indistinct edges.
    """
    detection = _detection_copy(image)
    blurred = cv2.GaussianBlur(detection, (5, 5), 0)
    edged   = cv2.Canny(blurred, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged  = cv2.dilate(edged, kernel, iterations=1)

    return _quad_from_mask(edged)


def detect_document(
    image: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Detect the document boundary using two strategies in order:

      1. GrabCut (colour-based foreground segmentation) — robust to
         crumpled edges because it segments by colour rather than geometry.
         Works well whenever the document is significantly lighter than
         the background (white receipt on dark table).

      2. Canny + contour (edge-based) — faster but fails when crumple
         creases fragment the document boundary into many short edges that
         never close into a clean quadrilateral.

    Returns a perspective-corrected crop of the document and the residual
    skew angle derived from the contour's top edge, or (None, 0.0) if both
    strategies fail.
    """
    for strategy_name, strategy_fn in (
        ("GrabCut", _detect_via_grabcut),
        ("Canny",   _detect_via_canny),
    ):
        pts = strategy_fn(image)
        if pts is not None:
            ordered = _order_points(pts)
            skew    = _skew_from_corners(ordered)
            cropped = _four_point_transform(image, pts)
            print(f"  [s1] Document detected via {strategy_name}")
            return cropped, skew

    return None, 0.0


# ── 1c. Coarse orientation via Tesseract OSD ──────────────────────────────────

def _coarse_orientation(image: np.ndarray) -> int:
    """
    Use Tesseract's OSD to detect coarse rotation (0, 90, 180, 270 °).
    Returns the angle to rotate *clockwise* to make text upright.
    Falls back to 0 if OSD fails (e.g. too little text detected).
    """
    pil = Image.fromarray(image)
    try:
        osd = pytesseract.image_to_osd(pil, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        return angle
    except pytesseract.TesseractError:
        return 0


def _rotate_image(image: np.ndarray, angle_cw: int) -> np.ndarray:
    """Rotate *image* clockwise by *angle_cw* degrees (must be 0/90/180/270)."""
    if angle_cw == 0:
        return image
    rotations = {90: cv2.ROTATE_90_CLOCKWISE,
                 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    code = rotations.get(angle_cw % 360)
    return cv2.rotate(image, code) if code is not None else image


def _apply_fine_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate *image* by *angle_deg* (counter-clockwise) around its centre,
    expanding the canvas so no content is clipped.  Background fill is white.
    """
    if abs(angle_deg) < 0.1:
        return image

    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Compute new bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ── 1e. Add background padding ───────────────────────────────────────────────

def _add_padding(image: np.ndarray, fraction: float = PADDING_FRACTION) -> np.ndarray:
    """
    Surround *image* with a white border of *fraction* × document_width on
    each side, simulating a document photographed with some background visible.
    This is important: UVDoc was trained on photos of documents with visible
    background, not tight crops.
    """
    h, w = image.shape[:2]
    pad = int(w * fraction)
    return cv2.copyMakeBorder(
        image, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


# ── Public Stage 1 entry point ────────────────────────────────────────────────

def preprocess_image(image_path: str, save_debug: bool = False,
                     debug_dir: str = "debug") -> Image.Image:
    """
    Stage 1: load → detect document → perspective correct →
             coarse orientation → fine skew (from contour) → pad.

    Returns a PIL RGB image of the upright document with white padding,
    ready for Stage 2 tiling.
    """
    stem = Path(image_path).stem
    if save_debug:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    print(f"  [s1] Loaded {image.shape[1]}×{image.shape[0]}")

    # 1b – document detection + perspective correction
    # The skew angle is derived from the contour's top edge here, not
    # from a separate Hough pass on the cropped image.
    cropped, contour_skew = detect_document(image)
    if cropped is None:
        print("  [s1] Warning: document not detected, using full image")
        cropped = image
        contour_skew = 0.0
    else:
        print(f"  [s1] Document detected → {cropped.shape[1]}×{cropped.shape[0]}, "
              f"contour skew {contour_skew:.2f}°")

    if save_debug:
        cv2.imwrite(f"{debug_dir}/{stem}_1b_cropped.png",
                    cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    # 1c – coarse orientation (0 / 90 / 180 / 270 °) via Tesseract OSD
    coarse = _coarse_orientation(cropped)
    if coarse:
        print(f"  [s1] Coarse rotation: {coarse}°")
        cropped = _rotate_image(cropped, coarse)

    # 1d – fine skew correction using the angle extracted from the contour.
    # We deliberately do NOT run Hough on the cropped image: crumple lines
    # and interior text lines dominate the Hough vote and can worsen the tilt.
    if abs(contour_skew) >= 0.1:
        print(f"  [s1] Fine skew correction: {contour_skew:.2f}°")
        cropped = _apply_fine_rotation(cropped, -contour_skew)

    if save_debug:
        cv2.imwrite(f"{debug_dir}/{stem}_1d_oriented.png",
                    cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    # 1e – padding
    padded = _add_padding(cropped)
    print(f"  [s1] After padding: {padded.shape[1]}×{padded.shape[0]}")

    if save_debug:
        cv2.imwrite(f"{debug_dir}/{stem}_1e_padded.png",
                    cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))

    return Image.fromarray(padded)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – tiling
# ═══════════════════════════════════════════════════════════════════════════════

def _doc_region(image: Image.Image,
                pad_fraction: float = PADDING_FRACTION) -> Tuple[int, int, int, int]:
    """
    Return the (x0, y0, x1, y1) bounding box of the document region inside
    the padded image output from Stage 1.

    Because Stage 1 adds a fixed *pad_fraction* of the *pre-padding* document
    width on all sides, we can recover the document bounds exactly.

    padded_width  = doc_width  + 2 * pad
    pad           = doc_width  * pad_fraction
    → doc_width   = padded_width / (1 + 2 * pad_fraction)
    """
    pw, ph = image.size
    doc_w = pw / (1 + 2 * pad_fraction)
    doc_h = ph / (1 + 2 * pad_fraction)
    pad_x = int((pw - doc_w) / 2)
    pad_y = int((ph - doc_h) / 2)
    return pad_x, pad_y, pw - pad_x, ph - pad_y


def compute_tiles(
    image: Image.Image,
    a4_aspect: float = A4_ASPECT,
    overlap_fraction: float = OVERLAP_FRACTION,
    pad_fraction: float = PADDING_FRACTION,
) -> List[Tile]:
    """
    Slice the Stage-1 output into tiles such that:

      • The document portion of each tile has an A4 aspect ratio
        (tile_height is derived from *document_width*, not padded_width)
      • Consecutive tiles overlap by ≥ *overlap_fraction* of tile height
      • Each tile is padded top/bottom with white so the document slice
        sits centred on a white canvas — the tile's overall dimensions
        are (padded_image_width × tile_height_with_padding), which makes
        it look like a photo of an A4 document taken at a distance

    Key distinction
    ───────────────
    Using `doc_width * A4_aspect` (not `image_width * A4_aspect`) ensures
    the *visible document content* inside each tile truly has A4 proportions,
    matching UVDoc's training distribution.
    """
    img_w, img_h = image.size
    x0, y0, x1, y1 = _doc_region(image, pad_fraction)

    doc_w = x1 - x0          # document width (pixels, excluding side padding)
    doc_h = y1 - y0          # document height (pixels, excluding top/bottom padding)

    # Height of the document content shown in one tile
    doc_tile_h = int(doc_w * a4_aspect)

    # Overlap in document-content pixels
    overlap = max(1, int(doc_tile_h * overlap_fraction))
    step = doc_tile_h - overlap

    # The side padding (left of document in the padded image) stays constant.
    # Top/bottom padding per tile is computed to centre the document slice.
    side_pad = x0      # same as y0 for a square padding, but we use x0 explicitly

    tiles: List[Tile] = []
    doc_y = 0          # current position inside the document region
    idx = 0

    while doc_y < doc_h:
        doc_y_end = min(doc_y + doc_tile_h, doc_h)
        slice_h = doc_y_end - doc_y     # may be < doc_tile_h for the last tile

        # Absolute rows in the padded image
        abs_y_start = y0 + doc_y
        abs_y_end   = y0 + doc_y_end

        # Crop the full width of the padded image so side background is included
        crop = image.crop((0, abs_y_start, img_w, abs_y_end))

        # Pad top/bottom so the document slice is centred on an A4 canvas.
        # The target tile height: side_pad on each side + doc_tile_h
        target_tile_h = doc_tile_h + 2 * side_pad

        # How much padding does this slice need?
        total_v_pad = target_tile_h - crop.height
        pad_top    = total_v_pad // 2
        pad_bottom = total_v_pad - pad_top

        canvas = Image.new("RGB", (img_w, target_tile_h), color=(255, 255, 255))
        canvas.paste(crop, (0, pad_top))

        tiles.append(Tile(
            image=canvas,
            index=idx,
            doc_y_start=doc_y,
            doc_y_end=doc_y_end,
            padded_top=pad_top,
            padded_bottom=pad_bottom,
        ))
        idx += 1

        if doc_y_end == doc_h:
            break
        doc_y += step

    return tiles


# ── Public Stage 2 entry point ────────────────────────────────────────────────

def tile_image(preprocessed: Image.Image, save_debug: bool = False,
               debug_dir: str = "debug", stem: str = "image") -> List[Tile]:
    """
    Stage 2: slice the Stage-1 output into A4-proportioned tiles with overlap.
    """
    tiles = compute_tiles(preprocessed)
    doc_tile_h = tiles[0].image.height if tiles else 0
    print(f"  [s2] {len(tiles)} tile(s), "
          f"each {tiles[0].image.width}×{doc_tile_h}px, "
          f"overlap {OVERLAP_FRACTION:.0%}")

    if save_debug:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        for t in tiles:
            t.image.save(f"{debug_dir}/{stem}_tile_{t.index:02d}.png")

    return tiles


# ── CLI / quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(pytesseract.get_languages(config=''))
    
    """

    path = "../../testing-materials/receipt_deu_02.jpg"
    #stem = Path(path).stem

    print("=== Stage 1: preprocessing ===")
    preprocessed = preprocess_image(path, save_debug=True)
    print(f"  Output: {preprocessed.size[0]}x{preprocessed.size[1]}")

    print("\n=== Stage 2: tiling ===")
    tiles = tile_image(preprocessed, save_debug=True)

    print(f"\nDone. {len(tiles)} tile(s) written to debug/")
    for t in tiles:
        w, h = t.image.size
        print(f"  tile {t.index:02d}  {w}x{h}  "
              f"doc rows {t.doc_y_start}-{t.doc_y_end}  "
              f"pad ↑{t.padded_top} ↓{t.padded_bottom}")
    """