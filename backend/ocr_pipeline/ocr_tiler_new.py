"""
ocr_pipeline/ocr_tiler.py
=========================
Produces UVDoc-ready tiles from a receipt photo to be used in ocr_dewarper.

Pipeline:
    1. Detect receipt via projection-guided contour search + minAreaRect
        NOTE: should correctly handle arbitrary rotation angles, if it doesn't then gg
    2. Center deskewed receipt on a background-coloured canvas
    3. Slice into A4-proportioned tiles (top / middle / bottom) with overlap
    4. Pad each tile symmetrically so the receipt section sits centered like an A4 document in the middle of the frame

Usage:
    import cv2, pytesseract
    from ocr_pipeline.ocr_tiler import get_tiles

    image = cv2.imread("receipt.jpg")
    texts = []

    for tile in get_tiles(image):
        rgb  = cv2.cvtColor(tile["image"], cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb, lang="deu")
        texts.append(text)

    full_text = "\n".join(texts)
"""

import argparse
import os
import warnings
from pathlib import Path

import cv2
import numpy as np


TARGET_RATIO = 210 / 297   # A4 document 


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sample_background_color(image: np.ndarray) -> np.ndarray:
    """Median colour of the four image corners"""
    h, w = image.shape[:2]
    s = max(20, min(h, w) // 30)
    corners = [image[:s, :s], image[:s, -s:], image[-s:, :s], image[-s:, -s:]]
    samples = np.vstack([c.reshape(-1, 3) for c in corners])
    return np.median(samples, axis=0).astype(np.uint8)


def order_and_warp(image: np.ndarray, rect, bg_scalar: tuple) -> np.ndarray:
    """Given a cv2.minAreaRect, perspective-warp the receipt to a straight portrait image"""
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
# Step 1 - Detection & deskew
# ---------------------------------------------------------------------------

def detect_and_deskew(
    image: np.ndarray,
    debug_dir: str | None = None,
    stem: str = "img",
) -> tuple[np.ndarray, np.ndarray, float]:
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_color = sample_background_color(image)
    h_img, w_img = image.shape[:2]

    # --- Threshold (unchanged) ---
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    forced = float(np.clip(otsu_val, 100, 180))
    _, thresh = cv2.threshold(gray, forced, 255, cv2.THRESH_BINARY)

    # --- Projection-based crop (unchanged) ---
    col_proj = thresh.mean(axis=0)
    row_proj = thresh.mean(axis=1)
    col_th   = max(10.0, col_proj.max() * 0.10)
    row_th   = max(10.0, row_proj.max() * 0.10)
    col_mask = col_proj > col_th
    row_mask = row_proj > row_th

    if col_mask.any() and row_mask.any():
        left   = int(np.argmax(col_mask))
        right  = int(w_img - np.argmax(col_mask[::-1]) - 1)
        top    = int(np.argmax(row_mask))
        bottom = int(h_img - np.argmax(row_mask[::-1]) - 1)
    else:
        left, right, top, bottom = 0, w_img - 1, 0, h_img - 1

    marg = max(50, int(min(w_img, h_img) * 0.03))
    cx0  = max(0, left  - marg);  cx1 = min(w_img, right  + marg)
    cy0  = max(0, top   - marg);  cy1 = min(h_img, bottom + marg)

    crop_thresh = thresh[cy0:cy1, cx0:cx1]

    # --- Morphological close (unchanged) ---
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 60))
    k_sq   = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(crop_thresh, cv2.MORPH_CLOSE, k_vert, iterations=2)
    closed = cv2.morphologyEx(closed,      cv2.MORPH_CLOSE, k_sq,   iterations=3)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_thresh.jpg"),
                    cv2.resize(closed, None, fx=0.5, fy=0.5))

    # --- Contour selection (unchanged) ---
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
        short = min(rw, rh); long_ = max(rw, rh)
        aspect = short / long_
        score  = area * (1.0 - aspect * 0.5)
        if score > best_score:
            best_score = score
            best = (cnt, rect)

    if best is None:
        warnings.warn("Receipt detection failed; using full image.")
        return image.copy(), bg_color, 0.0

    cnt, rect_crop = best
    cx, cy = rect_crop[0]
    rw, rh = rect_crop[1]
    angle  = rect_crop[2]

    # --- Derive rotation angle (unchanged) ---
    rot_angle = angle
    if rw > rh:
        rot_angle += 90.0

    # --- Rotate full image (unchanged) ---
    img_cx, img_cy = w_img / 2.0, h_img / 2.0
    M = cv2.getRotationMatrix2D((img_cx, img_cy), rot_angle, scale=1.0)
    cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
    new_w = int(h_img * sin_a + w_img * cos_a)
    new_h = int(h_img * cos_a + w_img * sin_a)
    M[0, 2] += (new_w / 2.0) - img_cx
    M[1, 2] += (new_h / 2.0) - img_cy

    bg_scalar = tuple(int(c) for c in bg_color)
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_scalar,
    )

    # --- FIX: crop the rotated image to the receipt bounds ---
    # Transform the four corners of the detected receipt rect into the
    # rotated image's coordinate space, then take their bounding box.
    # rect_crop is in crop-space; translate back to full-image space first.
    rect_full = ((cx + cx0, cy + cy0), rect_crop[1], rect_crop[2])
    box_full  = cv2.boxPoints(rect_full).astype(np.float32)          # 4×2, original coords
    box_rot   = (M[:, :2] @ box_full.T).T + M[:, 2]                  # apply affine to each point

    x0 = max(0, int(np.floor(box_rot[:, 0].min())))
    y0 = max(0, int(np.floor(box_rot[:, 1].min())))
    x1 = min(new_w, int(np.ceil(box_rot[:, 0].max())))
    y1 = min(new_h, int(np.ceil(box_rot[:, 1].max())))

    # Enforce portrait orientation of the crop
    if (x1 - x0) > (y1 - y0):
        x0, y0, x1, y1 = y0, x0, y1, x1   # swap axes if landscape

    receipt_crop = rotated[y0:y1, x0:x1]

    if debug_dir:
        box = cv2.boxPoints(rect_full).astype(np.int32)
        dbg = image.copy()
        cv2.drawContours(dbg, [box], 0, (0, 255, 0), 5)
        cv2.rectangle(dbg, (cx0, cy0), (cx1, cy1), (0, 165, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_debug_contour.jpg"), dbg)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_rotated.jpg"),
                    cv2.resize(rotated, None, fx=0.5, fy=0.5))
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_cropped.jpg"),
                    cv2.resize(receipt_crop, None, fx=0.5, fy=0.5))
    # After rotation, enforce portrait on the rect dimensions too
    actual_rw = min(rw, rh)
    actual_rh = max(rw, rh)
    return rotated, bg_color, rot_angle, (int(actual_rw), int(actual_rh))


# ---------------------------------------------------------------------------
# Step 2 - Center on background canvas
# ---------------------------------------------------------------------------

def center_receipt_on_background(
    receipt: np.ndarray,
    bg_color: np.ndarray,
    side_pad_frac: float = 0.12,
    vert_pad_frac: float = 0.04,
) -> tuple[np.ndarray, int, int]:
    """
    Embed receipt in a background canvas.

    CHANGED: Instead of filling the padding with a flat bg_color, the canvas
    is built by mirroring the actual edge pixels of the rotated image so that
    the padding blends naturally with whatever surface the receipt was
    photographed on.  This avoids a hard colour border that could confuse UVDoc.

    Returns (padded_image, pad_x, pad_y).
    Vertical padding is intentionally small - tiling handles the vertical axis.
    """
    h, w  = receipt.shape[:2]
    pad_x = int(w * side_pad_frac)
    pad_y = int(h * vert_pad_frac)

    new_h = h + 2 * pad_y
    new_w = w + 2 * pad_x

    # CHANGED: Start with a flat bg_color canvas as a safe fallback, then
    # overwrite padding regions with reflected edge content so the background
    # looks continuous rather than artificially solid.
    canvas = np.full((new_h, new_w, 3), bg_color, dtype=np.uint8)

    # Place the receipt in the centre first.
    canvas[pad_y: pad_y + h, pad_x: pad_x + w] = receipt

    if pad_x > 0:
        # CHANGED: Left pad — mirror the left edge strip of the receipt.
        left_strip  = receipt[:, :pad_x, :][:, ::-1, :]   # flip horizontally
        canvas[pad_y: pad_y + h, :pad_x, :] = left_strip

        # CHANGED: Right pad — mirror the right edge strip of the receipt.
        right_strip = receipt[:, -pad_x:, :][:, ::-1, :]
        canvas[pad_y: pad_y + h, pad_x + w:, :] = right_strip

    if pad_y > 0:
        # Top pad — mirror the top edge rows of the receipt itself (full canvas
        # width is handled by blitting the already-placed left/right strips).
        # Reading from `receipt` directly (not the canvas) avoids accidentally
        # sampling the canvas padding zone when pad_y >= h.
        top_src   = receipt[:pad_y, :, :][::-1, :, :]          # flip vertically
        canvas[pad_y - len(top_src): pad_y, pad_x: pad_x + w, :] = top_src

        # Bottom pad — mirror the bottom edge rows of the receipt.
        bottom_src = receipt[h - pad_y:, :, :][::-1, :, :]
        canvas[pad_y + h: pad_y + h + len(bottom_src), pad_x: pad_x + w, :] = bottom_src

    return canvas, pad_x, pad_y


# ---------------------------------------------------------------------------
# Step 3 - Tiling
# ---------------------------------------------------------------------------

def compute_tiles(
    receipt_w: int,
    receipt_h: int,
    overlap:   float = 0.15,
) -> list[dict]:
    """
    Divide receipt height into A4-proportioned slices

    Each tile's receipt window: width = receipt_w, height = receipt_w / TARGET_RATIO.
    Step between tiles   = tile_h x (1 - overlap).

    Returns list of dicts: {type, ry_start, ry_end}
    """
    tile_rh = int(round(receipt_w / TARGET_RATIO))
    step    = max(1, int(round(tile_rh * (1.0 - overlap))))

    # If the receipt fits in a single tile, one tile covers everything.
    if receipt_h <= tile_rh:
        return [{"index": 0, "type": "top", "ry_start": 0, "ry_end": tile_rh}]

    # Collect tile start positions using the fixed step.
    starts = []
    ry = 0
    while True:
        starts.append(ry)
        next_ry = ry + step
        # If the next tile's end exceeds the bottom of the receipt,
        # shift this tile back so it ends exactly at receipt_h to
        # guarantee TARGET_RATIO proportions
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
        tiles.append({"index": i, "type": t_type, "ry_start": s, "ry_end": s + tile_rh})
    return tiles


# ---------------------------------------------------------------------------
# Step 4 - Tile extraction with symmetric padding
# ---------------------------------------------------------------------------

def extract_tile(
    padded: np.ndarray,
    tile:   dict,
    rx: int, ry: int,   # receipt top-left in padded image
    rw: int,            # receipt width
) -> np.ndarray:
    """
    Crop the tile region and pad so the A4 receipt window sits centered
    with equal margins on all four sides (margin = rx = side padding).

    top    - keeps natural BG above the receipt, pads below
    bottom - keeps natural BG below the receipt, pads above
    middle - pads both top and bottom equally

    CHANGED: The canvas is filled with neutral mid-grey (128, 128, 128) rather
    than bg_color.  UVDoc relies on contrast between the document surface and
    the surrounding background to locate the document boundary; using the same
    colour as the receipt background removes that contrast and causes detection
    failures.  Mid-grey is visually neutral, guaranteed to differ from both
    bright-white receipts and dark-surface backgrounds, and does not carry any
    receipt texture that could mislead the dewarper.
    """
    img_h, img_w = padded.shape[:2]
    tile_rh      = tile["ry_end"] - tile["ry_start"]
    margin       = rx                   # equal margin target

    target_h = tile_rh + 2 * margin
    target_w = rw      + 2 * margin
    # CHANGED: use neutral mid-grey fill instead of bg_color for UVDoc contrast.
    uvdoc_fill = (128, 128, 128)
    canvas   = np.full((target_h, target_w, 3), uvdoc_fill, dtype=np.uint8)

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
    """Upscale to >1240 px wide, CLAHE, bilateral filter, unsharp mask."""
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

def get_tiles(
    image: np.ndarray,
    overlap: float = 0.15,  # percentage of overlap between tiles
) -> list[dict]:
    """
    Core pipeline. Accepts a BGR numpy array and returns tile dicts in order
    (top → middle* → bottom), each containing:

        {
            "type":  "top" | "middle" | "bottom",
            "index": int,         
            "image": np.ndarray,   # tile ready for OCR
        }
    """
    receipt, bg_color, _, (r_w, r_h) = detect_and_deskew(image)
    # receipt = enhance_for_ocr(receipt)
    padded, pad_x, pad_y = center_receipt_on_background(receipt, bg_color)
    # r_h, r_w = receipt.shape[:2]

    result = []
    for tile in compute_tiles(r_w, r_h, overlap=overlap):
        t   = tile["type"]
        idx = tile["index"]
        result.append({
            "index": idx,
            "type":  t,
            "image": extract_tile(padded, tile, pad_x, pad_y, r_w)
        })
    return result


# ---------------------------------------------------------------------------
# Disk-saving wrapper (CLI / batch use)
# ---------------------------------------------------------------------------

def save_tiles(
    input_path: str,
    out_dir:    str   = "tiles",
    overlap:    float = 0.15,
    debug:      bool  = False,
) -> list[str]:
    """Save tiles to disk. Returns list of written file paths."""
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(input_path).stem

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")

    if debug:
        receipt, bg_color, angle = detect_and_deskew(image, debug_dir=out_dir, stem=stem)
        print(f"  Rotated: {receipt.shape[1]}x{receipt.shape[0]}  "
              f"(angle corrected: {angle:.1f}°)")
        padded, pad_x, pad_y = center_receipt_on_background(receipt, bg_color)
        r_h, r_w = receipt.shape[:2]
        dbg_pad = padded.copy()
        cv2.rectangle(dbg_pad, (pad_x, pad_y),
                      (pad_x + r_w, pad_y + r_h), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_rotated.jpg"), receipt)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_padded.jpg"), dbg_pad)

    saved = []
    for t in get_tiles(image, overlap=overlap):
        fname = f"{stem}_tile_{t['type']}_{t['index']:02d}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, t["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(fpath)
        print(f"  Saved {fname}  [{t['image'].shape[1]}x{t['image'].shape[0]}]")
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__": 
    p = argparse.ArgumentParser(description="Receipt OCR preprocessing pipeline v3")
    p.add_argument("input")
    p.add_argument("--out_dir",  default="tiles")
    p.add_argument("--overlap",  type=float, default=0.15)
    p.add_argument("--debug",    action="store_true")
    args = p.parse_args()

    print(f"Processing: {args.input}")
    paths = save_tiles(args.input, args.out_dir, args.overlap, args.debug)
    print(f"\nDone - {len(paths)} tile(s) written to '{args.out_dir}/'")