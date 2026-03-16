"""
THE PLAN


Image preprocessing: 
1. Increase contrast for better edge detection on the document 
2. Center the document, orient it so the top edge is straight, and pad edges 
3. Output: image with document in the center with some padding on the sides

Image tiling: 
1. Given the preprocessing output, create tiles such that the document part of the tiles have an A4 aspect ratio. There should at least be 15% overlap 
2. After creating the tiles, add padding to the top or bottom so that the cut document resembles a centered A4 paper 
3. Output: multiple tiles with the document part of the original image, each resembling papers with A4 ratios

Tile dewarping: 
1. With each tile, apply the unwarping script 
2. Output: multiple tiles with rectified papers

Tile preprocessing: 
1. With the tiles, apply preprocessing steps to prepare for OCR 
2. Output: the preprocessed tiles

OCR process: 
1. With the preprocessed tiles, apply OCR to extract the text 
2. Merge the extracted texts into one big text at the overlaps 
3. Output: the final text
"""

"""
Receipt OCR Preprocessing Pipeline  (v3)
=========================================
Produces UVDoc-ready tiles from a receipt photo.

Pipeline:
  1. Detect receipt via projection-guided contour search + minAreaRect
     → correctly handles arbitrary rotation angles
  2. Centre deskewed receipt on a background-coloured canvas
  3. Slice into A4-proportioned tiles (top / middle / bottom) with overlap
  4. Pad each tile symmetrically so the receipt section sits centred
     like an A4 document in the middle of the frame

Fixes over v1/v2:
  - Uses column/row projection to find the rough receipt band first, then
    runs contour detection only within that crop — avoids false-positive
    background contours that are too wide/square.
  - Aspect filter relaxed to 0.85 with a secondary score check.
  - minAreaRect-based warp replaces approxPolyDP (handles non-90° angles).

Usage:
    python receipt_preprocess.py <input_image> [--out_dir tiles] [--overlap 0.15] [--debug]
"""

import argparse
import os
import warnings
from pathlib import Path

import cv2
import numpy as np

A4_RATIO = 210 / 297   # width / height


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_background_color(image: np.ndarray) -> np.ndarray:
    """Median colour of the four image corners (robust to bright receipt)."""
    h, w = image.shape[:2]
    s = max(20, min(h, w) // 30)
    corners = [image[:s, :s], image[:s, -s:], image[-s:, :s], image[-s:, -s:]]
    samples = np.vstack([c.reshape(-1, 3) for c in corners])
    return np.median(samples, axis=0).astype(np.uint8)


def order_and_warp(image: np.ndarray, rect, bg_scalar: tuple) -> np.ndarray:
    """
    Given a cv2.minAreaRect, perspective-warp the receipt to a
    straight portrait image.
    """
    box = cv2.boxPoints(rect).astype(np.float32)
    s    = box.sum(axis=1)
    diff = np.diff(box, axis=1).flatten()
    tl   = box[np.argmin(s)]
    br   = box[np.argmax(s)]
    tr   = box[np.argmin(diff)]
    bl   = box[np.argmax(diff)]
    pts_src = np.array([tl, tr, br, bl], dtype=np.float32)

    w_r = int(round(np.linalg.norm(tr - tl)))
    h_r = int(round(np.linalg.norm(bl - tl)))

    # Enforce portrait
    if w_r > h_r:
        w_r, h_r = h_r, w_r
        pts_src = np.roll(pts_src, 1, axis=0)

    pts_dst = np.array(
        [[0, 0], [w_r - 1, 0], [w_r - 1, h_r - 1], [0, h_r - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(
        image, M, (w_r, h_r),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_scalar,
    )


# ---------------------------------------------------------------------------
# Step 1 – Detection & deskew
# ---------------------------------------------------------------------------

def detect_and_deskew(
    image: np.ndarray,
    debug_dir: str | None = None,
    stem: str = "img",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Detect the receipt boundary and correct perspective / rotation.

    Strategy:
    1.  Threshold → column & row projection to find rough receipt band.
    2.  Crop to that band (+ margin) and run morphological close inside
        the crop — avoids bridging to bright areas outside the receipt.
    3.  Find the largest contour and fit minAreaRect.
    4.  Perspective-warp to a straight portrait image.

    Returns (deskewed_image, bg_color, angle_corrected).
    """
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_color = sample_background_color(image)
    h_img, w_img = image.shape[:2]

    # ── Threshold ────────────────────────────────────────────────────────────
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    forced = float(np.clip(otsu_val, 100, 180))
    _, thresh = cv2.threshold(gray, forced, 255, cv2.THRESH_BINARY)

    # ── Projection-based crop ─────────────────────────────────────────────
    # Column projection: find the horizontal band containing the receipt
    col_proj  = thresh.mean(axis=0)
    row_proj  = thresh.mean(axis=1)

    # Receipt is the dominant bright band; threshold at 10% of max brightness
    col_th    = max(10.0, col_proj.max() * 0.10)
    row_th    = max(10.0, row_proj.max() * 0.10)

    col_mask  = col_proj > col_th
    row_mask  = row_proj > row_th

    if col_mask.any() and row_mask.any():
        left   = int(np.argmax(col_mask))
        right  = int(w_img - np.argmax(col_mask[::-1]) - 1)
        top    = int(np.argmax(row_mask))
        bottom = int(h_img - np.argmax(row_mask[::-1]) - 1)
    else:
        left, right, top, bottom = 0, w_img - 1, 0, h_img - 1

    marg   = max(50, int(min(w_img, h_img) * 0.03))
    cx0    = max(0, left  - marg)
    cx1    = min(w_img, right  + marg)
    cy0    = max(0, top   - marg)
    cy1    = min(h_img, bottom + marg)

    crop_thresh = thresh[cy0:cy1, cx0:cx1]

    # ── Morphological close inside crop ──────────────────────────────────
    # Tall vertical kernel to join the receipt body across horizontal text gaps
    k_vert  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 60))
    k_sq    = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed  = cv2.morphologyEx(crop_thresh, cv2.MORPH_CLOSE, k_vert, iterations=2)
    closed  = cv2.morphologyEx(closed,      cv2.MORPH_CLOSE, k_sq,   iterations=3)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_thresh.jpg"),
                    cv2.resize(closed, None, fx=0.5, fy=0.5))

    # ── Contour selection ─────────────────────────────────────────────────
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = (cx1 - cx0) * (cy1 - cy0)

    best, best_score = None, -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.05 * crop_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        short = min(rw, rh);  long = max(rw, rh)
        aspect = short / long
        # Accept everything up to almost-square; score prefers narrow+tall
        score  = area * (1.0 - aspect * 0.5)
        if score > best_score:
            best_score = score
            best = (cnt, rect)

    if best is None:
        warnings.warn("Receipt detection failed; using full image.")
        h, w = image.shape[:2]
        fake_rect = ((w / 2, h / 2), (float(w), float(h)), 0.0)
        return image.copy(), bg_color, 0.0

    cnt, rect_crop = best
    # Translate rect centre back to full-image coordinates
    cx, cy  = rect_crop[0]
    size    = rect_crop[1]
    angle   = rect_crop[2]
    rect_full = ((cx + cx0, cy + cy0), size, angle)

    bg_scalar = tuple(int(c) for c in bg_color)
    deskewed  = order_and_warp(image, rect_full, bg_scalar)

    if debug_dir:
        box = cv2.boxPoints(rect_full).astype(np.int32)
        dbg = image.copy()
        cv2.drawContours(dbg, [box], 0, (0, 255, 0), 5)
        # Also draw projection bounds
        cv2.rectangle(dbg, (cx0, cy0), (cx1, cy1), (0, 165, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_debug_contour.jpg"), dbg)

    return deskewed, bg_color, angle


# ---------------------------------------------------------------------------
# Step 2 – Centre on background canvas
# ---------------------------------------------------------------------------

def centre_receipt_on_background(
    receipt: np.ndarray,
    bg_color: np.ndarray,
    side_pad_frac: float = 0.12,
    vert_pad_frac: float = 0.04,
) -> tuple[np.ndarray, int, int]:
    """
    Embed receipt in a background canvas.

    Returns (padded_image, pad_x, pad_y).
    Vertical padding is intentionally small — tiling handles the vertical axis.
    """
    h, w     = receipt.shape[:2]
    pad_x    = int(w * side_pad_frac)
    pad_y    = int(h * vert_pad_frac)
    canvas   = np.full((h + 2 * pad_y, w + 2 * pad_x, 3), bg_color, dtype=np.uint8)
    canvas[pad_y: pad_y + h, pad_x: pad_x + w] = receipt
    return canvas, pad_x, pad_y


# ---------------------------------------------------------------------------
# Step 3 – Tiling
# ---------------------------------------------------------------------------

def compute_tiles(
    receipt_w: int,
    receipt_h: int,
    overlap:   float = 0.15,
) -> list[dict]:
    """
    Divide receipt height into A4-proportioned slices.

    Each tile's receipt window: width = receipt_w, height = receipt_w / A4_RATIO.
    Step between tiles   = tile_h × (1 − overlap).

    Returns list of dicts: {type, ry_start, ry_end}
    """
    tile_rh = int(round(receipt_w / A4_RATIO))
    step    = max(1, int(round(tile_rh * (1.0 - overlap))))

    # If the receipt fits in a single tile, one tile covers everything.
    if receipt_h <= tile_rh:
        return [{"type": "top", "ry_start": 0, "ry_end": tile_rh}]

    # Collect tile start positions using the fixed step.
    starts = []
    ry = 0
    while True:
        starts.append(ry)
        next_ry = ry + step
        # Would the *next* tile's end reach/exceed the receipt bottom?
        # If so, shift this next tile back so it ends exactly at receipt_h,
        # guaranteeing A4 proportions. The resulting overlap will be >= min_overlap.
        if next_ry + tile_rh >= receipt_h:
            last_start = receipt_h - tile_rh   # shift back to fill exactly
            if last_start > ry:                # avoid duplicate when already snug
                starts.append(last_start)
            break
        ry = next_ry

    # Assign types and build tile dicts
    tiles = []
    for i, s in enumerate(starts):
        if i == 0:
            t_type = "top"
        elif i == len(starts) - 1:
            t_type = "bottom"
        else:
            t_type = "middle"
        tiles.append({"type": t_type, "ry_start": s, "ry_end": s + tile_rh})
    return tiles


# ---------------------------------------------------------------------------
# Step 4 – Tile extraction with symmetric padding
# ---------------------------------------------------------------------------

def extract_tile(
    padded: np.ndarray,
    tile:   dict,
    rx: int, ry: int,   # receipt top-left in padded image
    rw: int,            # receipt width
    bg_color: np.ndarray,
) -> np.ndarray:
    """
    Crop the tile region and pad so the A4 receipt window sits centred
    with equal margins on all four sides (margin = rx = side padding).

    • top    – keeps natural BG above the receipt, pads below
    • bottom – keeps natural BG below the receipt, pads above
    • middle – pads both top and bottom equally
    """
    img_h, img_w = padded.shape[:2]
    tile_rh      = tile["ry_end"] - tile["ry_start"]
    margin       = rx                   # equal margin target

    target_h = tile_rh + 2 * margin
    target_w = rw      + 2 * margin
    canvas   = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    abs_ry0 = ry + tile["ry_start"]
    abs_ry1 = ry + tile["ry_end"]

    if tile["type"] == "top":
        nat_top = min(margin, ry)
        src_y0  = max(0, abs_ry0 - nat_top)
        src_y1  = min(img_h, abs_ry1)
        dst_y0  = margin - nat_top
    elif tile["type"] == "bottom":
        nat_bot = min(margin, img_h - abs_ry1)
        src_y0  = max(0, abs_ry0)
        src_y1  = min(img_h, abs_ry1 + nat_bot)
        dst_y0  = margin
    else:
        src_y0  = max(0, abs_ry0)
        src_y1  = min(img_h, abs_ry1)
        dst_y0  = margin

    src_x0 = max(0, rx - margin)
    src_x1 = min(img_w, rx + rw + margin)

    actual_h = src_y1 - src_y0
    actual_w = src_x1 - src_x0
    dst_y1   = dst_y0 + actual_h
    dst_x0   = margin - (rx - src_x0)
    dst_x1   = dst_x0 + actual_w

    dy0 = max(0, dst_y0); dy1 = min(target_h, dst_y1)
    dx0 = max(0, dst_x0); dx1 = min(target_w, dst_x1)
    sy0 = src_y0 + max(0, -dst_y0)
    sx0 = src_x0 + max(0, -dst_x0)
    sy1 = sy0 + (dy1 - dy0)
    sx1 = sx0 + (dx1 - dx0)

    if dy1 > dy0 and dx1 > dx0 and sy1 > sy0 and sx1 > sx0:
        canvas[dy0:dy1, dx0:dx1] = padded[sy0:sy1, sx0:sx1]
    return canvas


# ---------------------------------------------------------------------------
# OCR enhancements
# ---------------------------------------------------------------------------

def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Upscale to ≥1240 px wide, CLAHE, bilateral filter, unsharp mask."""
    h, w = image.shape[:2]
    if w < 1240:
        scale = 1240 / w
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_LANCZOS4)

    lab   = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    blur  = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    return image


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_receipt(
    input_path: str,
    out_dir:    str   = "tiles",
    overlap:    float = 0.15,
    debug:      bool  = False,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(input_path).stem

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")

    dbg = out_dir if debug else None

    # 1. Detect & deskew
    receipt, bg_color, angle = detect_and_deskew(image, debug_dir=dbg, stem=stem)
    print(f"  Deskewed: {receipt.shape[1]}×{receipt.shape[0]}  "
          f"(angle corrected: {angle:.1f}°)")
    if debug:
        cv2.imwrite(os.path.join(out_dir, f"{stem}_deskewed.jpg"), receipt)

    # OCR enhancement (upscale + CLAHE + sharpen)
    receipt = enhance_for_ocr(receipt)

    # 2. Centre on background
    padded, pad_x, pad_y = centre_receipt_on_background(receipt, bg_color)
    r_h, r_w = receipt.shape[:2]

    if debug:
        dbg_pad = padded.copy()
        cv2.rectangle(dbg_pad, (pad_x, pad_y),
                      (pad_x + r_w, pad_y + r_h), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_padded.jpg"), dbg_pad)

    # 3. Tile plan
    tiles   = compute_tiles(r_w, r_h, overlap=overlap)
    tile_rh = int(round(r_w / A4_RATIO))
    print(f"  Tiles: {len(tiles)}  |  A4 window per tile: {r_w}×{tile_rh} px")

    # 4. Extract + save
    saved:      list[str]      = []
    type_count: dict[str, int] = {}
    for tile in tiles:
        t   = tile["type"]
        idx = type_count.get(t, 0)
        type_count[t] = idx + 1

        tile_img = extract_tile(padded, tile, pad_x, pad_y, r_w, bg_color)
        fname = f"{stem}_tile_{t}_{idx:02d}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(fpath)
        print(f"  Saved {fname}  [{tile_img.shape[1]}×{tile_img.shape[0]}]")

    return saved


# ---------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for i in range(1, 8):
        PATH = f"../../testing-materials/receipt_deu_0{i}.jpg"
        """
        input_path: str,
        out_dir: str = "tiles",
        overlap: float = 0.15,
        debug: bool = False,
        """
        paths = process_receipt(PATH, "tiles", 0.20, False)
