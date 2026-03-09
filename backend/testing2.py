import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from unwarp import Unwarp

# ── Constants ────────────────────────────────────────────────────────────────

# UVDoc was trained at 488×712 → aspect ratio height/width ≈ 1.459
UVDOC_ASPECT = 712 / 488  # ≈ 1.459  (height / width)

# Overlap between tiles as a fraction of tile height.
# Large enough to guarantee shared text lines at boundaries,
# small enough to avoid excessive duplication.
OVERLAP_FRACTION = 0.15

# Minimum fraction of overlapping lines that must agree to be
# considered a valid join point during text merging.
MERGE_AGREEMENT_THRESHOLD = 0.6

# Minimum similarity score (0–1) for two lines to be considered the same.
LINE_SIMILARITY_THRESHOLD = 0.75

# How many lines back from the end of the merged text to search
# for the overlap join point.
MAX_OVERLAP_SEARCH = 15


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Tile:
    """A single vertical slice of the original image."""
    image: Image.Image      # cropped & padded tile ready for dewarping
    y_start: int            # first row in the original image (inclusive)
    y_end: int              # last row in the original image (exclusive)
    index: int
    padded_rows: int = 0    # rows of white padding added at the bottom


@dataclass
class DewarppedTile:
    tile: Tile
    image: Image.Image      # dewarped output (padding rows removed)
    text_lines: List[str] = field(default_factory=list)


# ── Stage 1 – Tiling ─────────────────────────────────────────────────────────

def compute_tiles(
    image: Image.Image,
    target_aspect: float = UVDOC_ASPECT,
    overlap_fraction: float = OVERLAP_FRACTION,
) -> List[Tile]:
    """
    Slice *image* into vertical tiles whose height/width ratio matches
    *target_aspect* so that UVDoc receives inputs close to its training
    distribution regardless of the original image's aspect ratio.

    Tiles overlap by *overlap_fraction* of tile height so that every
    text line near a boundary appears in two consecutive tiles — this
    gives the text-merge stage enough context to find the correct join
    point without relying on pixel-level stitching.

    A single tile covering the whole image is returned when the image
    aspect ratio is already within 20 % of *target_aspect*.
    """
    width, height = image.size
    current_aspect = height / width

    if current_aspect <= target_aspect * 1.20:
        return [Tile(image=image.copy(), y_start=0, y_end=height, index=0)]

    tile_h = int(width * target_aspect)
    overlap = int(tile_h * overlap_fraction)
    step = tile_h - overlap

    tiles: List[Tile] = []
    y = 0
    idx = 0

    while y < height:
        y_end = min(y + tile_h, height)
        crop = image.crop((0, y, width, y_end))

        # White-pad the last (shorter) tile so the model always receives
        # an image with the exact expected aspect ratio.
        padded_rows = 0
        if crop.height < tile_h:
            padded_rows = tile_h - crop.height
            padded = Image.new("RGB", (width, tile_h), color=(255, 255, 255))
            padded.paste(crop, (0, 0))
            crop = padded

        tiles.append(
            Tile(image=crop, y_start=y, y_end=y_end,
                 index=idx, padded_rows=padded_rows)
        )
        idx += 1

        if y_end == height:
            break
        y += step

    return tiles

# ADDED
def crop_to_document(image: Image.Image) -> Image.Image:
    """
    Detect the document/receipt outline and return a perspective-corrected
    crop containing only the document.
    Uses OpenCV's contour detection on a thresholded image.
    """
    import cv2
    import numpy as np

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the largest quadrilateral contour — that's the document
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return _four_point_transform(image, approx.reshape(4, 2))

    # Fallback: return the original if no quad found
    print("  [crop] Warning: no document outline detected, using full image")
    return image


def _four_point_transform(
    image: Image.Image, pts: np.ndarray
) -> Image.Image:
    """Perspective-correct crop given four corner points."""
    import cv2
    import numpy as np

    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    warped = cv2.warpPerspective(np.array(image), M, (width, height))
    return Image.fromarray(warped)


def _order_points(pts: np.ndarray) -> np.ndarray:
    import numpy as np
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[s.argmin()]   # top-left
    rect[2] = pts[s.argmax()]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[diff.argmin()]  # top-right
    rect[3] = pts[diff.argmax()]  # bottom-left
    return rect


# ── Stage 2 – Dewarping ───────────────────────────────────────────────────────

def dewarp_tiles(tiles: List[Tile], unwarper: Unwarp) -> List[DewarppedTile]:
    """
    Run the dewarper on every tile independently.

    Each tile is dewarped on its own merits — no pixel-level stitching is
    attempted later.  The overlap between tiles exists solely to provide
    redundant text lines for the merge stage.

    Padding rows added during tiling are cropped out of the dewarped output
    so that the last tile is not artificially taller than its content.
    """
    results: List[DewarppedTile] = []

    for tile in tiles:
        dewarped = unwarper.inference(tile.image)

        # Remove padding that was added to the bottom of the last tile.
        if tile.padded_rows > 0:
            keep_h = dewarped.height - tile.padded_rows
            dewarped = dewarped.crop((0, 0, dewarped.width, keep_h))

        results.append(DewarppedTile(tile=tile, image=dewarped))

    return results


# ── Stage 3 – OCR preprocessing ──────────────────────────────────────────────

def preprocess_for_ocr(image: Image.Image, min_width: int = 1000) -> Image.Image:
    """
    Greyscale → contrast boost → sharpen → upscale.

    Thermal-print receipts are low-contrast and often photographed at
    resolutions where Tesseract struggles.  These steps normalise the
    image without distorting character shapes.
    """
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)

    w, h = image.size
    if w < min_width:
        scale = math.ceil(min_width / w)
        image = image.resize((w * scale, h * scale), Image.LANCZOS)

    return image


# ── Stage 4 – Per-tile OCR ────────────────────────────────────────────────────

def ocr_tile(image: Image.Image, lang: str = "deu") -> List[str]:
    """
    Run Tesseract on a single dewarped tile and return a list of
    non-empty text lines.

    PSM 6 (uniform block of text) works well for receipt columns.
    No character whitelist is used — the language model handles
    character disambiguation better than a hard filter.
    """
    config = "--oem 3 --psm 6 --dpi 300"
    raw = pytesseract.image_to_string(image, lang=lang, config=config)
    return [line for line in raw.splitlines() if line.strip()]


def ocr_dewarped_tiles(
    dewarped_tiles: List[DewarppedTile],
    lang: str = "deu",
) -> List[DewarppedTile]:
    """
    OCR every dewarped tile independently and store the resulting lines
    on each DewarppedTile object.
    """
    for dt in dewarped_tiles:
        preprocessed = preprocess_for_ocr(dt.image)
        dt.text_lines = ocr_tile(preprocessed, lang=lang)

    return dewarped_tiles


# ── Stage 5 – Text merging ────────────────────────────────────────────────────

def _line_similarity(a: str, b: str) -> float:
    """
    Sequence-aware similarity between two normalised lines (0–1).

    SequenceMatcher is used in preference to character-level Jaccard
    because receipts contain many lines that share characters (prices,
    dates) but differ in order, e.g. "20.08" vs "08.20".
    """
    return SequenceMatcher(None, a, b).ratio()


def _normalize(line: str) -> str:
    return " ".join(line.lower().split())


def _find_overlap(
    merged: List[str],
    next_lines: List[str],
    max_search: int = MAX_OVERLAP_SEARCH,
    agreement_threshold: float = MERGE_AGREEMENT_THRESHOLD,
    similarity_threshold: float = LINE_SIMILARITY_THRESHOLD,
) -> int:
    """
    Find how many lines at the start of *next_lines* are already present
    at the end of *merged* (i.e. the overlapping region between two
    consecutive tiles).

    Returns the number of lines to skip from the start of *next_lines*
    when appending to *merged*.  Returns 0 if no reliable overlap is found,
    which means the next tile's lines are appended in full — a safe
    fallback that may produce a small number of duplicate lines but will
    never drop content.
    """
    merged_clean = [_normalize(l) for l in merged     if l.strip()]
    next_clean   = [_normalize(l) for l in next_lines if l.strip()]

    for k in range(min(max_search, len(merged_clean), len(next_clean)), 0, -1):
        tail = merged_clean[-k:]
        head = next_clean[:k]

        matches = sum(
            1 for a, b in zip(tail, head)
            if _line_similarity(a, b) >= similarity_threshold
        )

        if matches / k >= agreement_threshold:
            return k

    return 0


def merge_tile_texts(dewarped_tiles: List[DewarppedTile]) -> str:
    """
    Combine per-tile OCR results into a single coherent text by detecting
    and removing the duplicated lines that appear in the overlapping region
    between consecutive tiles.

    This approach is fundamentally more reliable than pixel stitching
    because it operates on recognised text rather than raw pixel geometry:
    non-linear warp corrections applied independently to adjacent tiles
    can shift the same physical text to different pixel positions, making
    seamless pixel stitching impossible to guarantee.  Text-level merging
    is unaffected by this since it compares recognised characters.
    """
    if not dewarped_tiles:
        return ""

    if len(dewarped_tiles) == 1:
        return "\n".join(dewarped_tiles[0].text_lines)

    merged: List[str] = [l for l in dewarped_tiles[0].text_lines if l.strip()]

    for dt in dewarped_tiles[1:]:
        next_lines = [l for l in dt.text_lines if l.strip()]

        if not next_lines:
            continue

        skip = _find_overlap(merged, next_lines)

        if skip == 0:
            # No reliable overlap found — append in full.
            # Duplicates are preferable to silently dropped content.
            print(
                f"  [merge] Warning: no overlap detected between tile "
                f"{dt.tile.index - 1} and {dt.tile.index}. "
                f"Appending tile {dt.tile.index} in full."
            )
        else:
            print(
                f"  [merge] Tile {dt.tile.index}: skipped {skip} overlapping "
                f"line(s), appended {len(next_lines) - skip} new line(s)"
            )

        merged.extend(next_lines[skip:])

    return "\n".join(merged)


# ── Public pipeline entry point ───────────────────────────────────────────────

def process_receipt(
    image_path: str,
    lang: str = "deu",
    save_debug: bool = False,
    debug_dir: str = "debug",
) -> str:
    """
    Full receipt OCR pipeline:

      1. Load image
      2. Slice into aspect-ratio-correct tiles (with overlap for merge context)
      3. Dewarp each tile independently with UVDoc
      4. Preprocess and OCR each tile independently with Tesseract
      5. Merge per-tile text by detecting and removing duplicate lines
         in the overlapping regions
      6. Return the combined text

    Pixel-level stitching is deliberately avoided: UVDoc applies a
    non-linear warp correction independently to each tile, meaning the
    same physical text line can land at different pixel positions in
    adjacent tiles after dewarping.  Merging at the text level side-steps
    this problem entirely.

    Parameters
    ----------
    image_path : str
        Path to the input receipt photo.
    lang : str
        Tesseract language string, e.g. ``"deu"``, ``"eng"``, ``"deu+eng"``.
    save_debug : bool
        If True, every intermediate image and OCR result is saved to
        *debug_dir* so that each stage can be inspected individually.
    debug_dir : str
        Directory for debug outputs (created automatically if absent).
    """
    image = Image.open(image_path).convert("RGB")
    stem  = Path(image_path).stem
    w, h  = image.size
    print(f"[1/5] Loaded  '{image_path}'  ({w}×{h}, aspect {h/w:.2f})")

    # TESTING
    image = crop_to_document(image)
    print(f"  [crop] Document crop: {image.size}")

    # ── 1. Tile ───────────────────────────────────────────────────────────────
    tiles = compute_tiles(image)
    print(f"[2/5] Tiled   → {len(tiles)} tile(s)  "
          f"(target aspect {UVDOC_ASPECT:.2f}, overlap {OVERLAP_FRACTION:.0%})")

    if save_debug:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        for t in tiles:
            t.image.save(f"{debug_dir}/{stem}_tile_{t.index:02d}.png")

    # ── 2. Dewarp ─────────────────────────────────────────────────────────────
    unwarper = Unwarp()
    dewarped = dewarp_tiles(tiles, unwarper)
    print(f"[3/5] Dewarped {len(dewarped)} tile(s) independently")

    if save_debug:
        for dt in dewarped:
            dt.image.save(f"{debug_dir}/{stem}_dewarped_{dt.tile.index:02d}.png")

    # ── 3. OCR per tile ───────────────────────────────────────────────────────
    dewarped = ocr_dewarped_tiles(dewarped, lang=lang)
    total_lines = sum(len(dt.text_lines) for dt in dewarped)
    print(f"[4/5] OCR'd   → {total_lines} raw lines across all tiles")

    if save_debug:
        for dt in dewarped:
            path = f"{debug_dir}/{stem}_ocr_{dt.tile.index:02d}.txt"
            Path(path).write_text("\n".join(dt.text_lines), encoding="utf-8")

    # ── 4. Merge ──────────────────────────────────────────────────────────────
    text = merge_tile_texts(dewarped)
    final_lines = len(text.splitlines())
    print(f"[5/5] Merged  → {final_lines} lines, {len(text)} chars")

    if save_debug:
        Path(f"{debug_dir}/{stem}_final.txt").write_text(text, encoding="utf-8")

    return text


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    IMAGE_PATH = "../testing-materials/receipt_deu_01.jpg"

    result = process_receipt(IMAGE_PATH, lang="deu", save_debug=True)

    print("\n" + "─" * 60)
    print(result)
    print("─" * 60)