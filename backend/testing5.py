"""
receipt_preprocess.py
─────────────────────
Preprocesses a receipt photo before feeding it into PaddleOCR.

What this does (and why):
    1. Detect the receipt boundary (GrabCut primary, Canny fallback)
    2. Perspective-correct crop via four-point transform
    3. Coarse orientation fix (0/90/180/270°) via Tesseract OSD
    4. Fine skew correction derived from the contour's top edge
    5. Center on a background-coloured canvas with padding

What this deliberately does NOT do:
    - Contrast enhancement / sharpening  → let PaddleOCR handle internally
    - Tiling                             → out of scope here

PaddleOCR settings to use alongside this script:
    use_doc_orientation_classify = False   (handled here via Tesseract OSD)
    use_doc_unwarping            = False   (handled here via perspective warp)
    use_textline_orientation     = True    (still useful per-line)

Dependencies: opencv-python, numpy, pytesseract
"""

import math
import warnings
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract


# ── Constants ─────────────────────────────────────────────────────────────────

PADDING_FRACTION = 0.05   # white border added around the corrected document


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[s.argmin()]     # top-left
    rect[2] = pts[s.argmax()]     # bottom-right
    rect[1] = pts[diff.argmin()]  # top-right
    rect[3] = pts[diff.argmax()]  # bottom-left
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-correct crop to the quadrilateral defined by pts."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image, M, (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _skew_from_corners(corners: np.ndarray) -> float:
    """
    Derive residual skew (degrees) from the top edge of the ordered contour.
    Using the document boundary rather than Hough lines avoids false votes
    from crumple creases and interior text.
    Positive → document tilted clockwise; negative → counter-clockwise.
    """
    tl, tr = corners[0], corners[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    if dx == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _quad_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest quadrilateral contour in a binary mask.
    Tries progressively looser epsilon values so crumpled edges still
    collapse to exactly 4 points.
    Returns float32 array of shape (4, 2), or None.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    image_area = mask.shape[0] * mask.shape[1]
    contours   = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:5]:
        if cv2.contourArea(c) < image_area * 0.10:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

    return None


# ── Document boundary detectors ───────────────────────────────────────────────

def _detect_via_grabcut(image: np.ndarray) -> Optional[np.ndarray]:
    """
    GrabCut (colour-based) — robust to crumpled edges because it segments
    by colour rather than geometry. Works well for a white receipt on a
    dark surface.
    """
    h, w = image.shape[:2]
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    gc_mask   = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model,
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    return _quad_from_mask(fg_mask)


def _detect_via_canny(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Canny (edge-based) fallback — faster but fails when crumple creases
    fragment the document boundary.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray    = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 30, 120)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged   = cv2.dilate(edged, kernel, iterations=1)
    return _quad_from_mask(edged)


def _detect_document(image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Try GrabCut then Canny. Returns (perspective-corrected crop, skew_angle)
    or (None, 0.0) if both strategies fail.
    """
    for name, fn in (("GrabCut", _detect_via_grabcut), ("Canny", _detect_via_canny)):
        pts = fn(image)
        if pts is not None:
            ordered = _order_points(pts)
            skew    = _skew_from_corners(ordered)
            cropped = _four_point_transform(image, pts)
            print(f"  [preprocess] Document detected via {name}")
            return cropped, skew

    return None, 0.0


# ── Orientation helpers ───────────────────────────────────────────────────────

def _coarse_orientation(image: np.ndarray) -> int:
    """
    Tesseract OSD coarse rotation detection (0 / 90 / 180 / 270°).
    Returns degrees to rotate clockwise to make text upright.
    Falls back to 0 if OSD fails.
    """
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        import PIL.Image
        pil = PIL.Image.fromarray(rgb)
        osd = pytesseract.image_to_osd(pil, output_type=pytesseract.Output.DICT)
        return int(osd.get("rotate", 0))
    except Exception:
        return 0


def _rotate_coarse(image: np.ndarray, angle_cw: int) -> np.ndarray:
    """Lossless 90° step rotation."""
    codes = {90: cv2.ROTATE_90_CLOCKWISE,
             180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    code = codes.get(angle_cw % 360)
    return cv2.rotate(image, code) if code is not None else image


def _rotate_fine(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Sub-degree affine rotation with canvas expansion so no content is clipped.
    Background filled white.
    """
    if abs(angle_deg) < 0.1:
        return image

    h, w   = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos    = abs(M[0, 0])
    sin    = abs(M[0, 1])
    new_w  = int(h * sin + w * cos)
    new_h  = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ── Padding ───────────────────────────────────────────────────────────────────

def _add_padding(image: np.ndarray,
                 fraction: float = PADDING_FRACTION) -> np.ndarray:
    """
    White border of (fraction × document_width) on all four sides.
    Gives PaddleOCR's detector a small margin around the document edges,
    which improves detection of text near the receipt boundary.
    """
    pad = int(image.shape[1] * fraction)
    return cv2.copyMakeBorder(
        image, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_for_ocr(
    image_path: str,
    save_debug: bool = False,
    debug_dir:  str  = "debug",
) -> np.ndarray:
    """
    Load a receipt photo and return a BGR numpy array ready for PaddleOCR.

    Steps applied:
        1. Document boundary detection + perspective correction
        2. Coarse orientation fix via Tesseract OSD (0/90/180/270°)
        3. Fine skew correction from contour top edge
        4. White padding around the document

    Parameters
    ----------
    image_path : str
        Path to the input receipt photo.
    save_debug : bool
        If True, intermediate images are saved to *debug_dir*.
    debug_dir : str
        Directory for debug images (created if needed).

    Returns
    -------
    np.ndarray
        BGR image ready to pass directly to ``ocr.predict()``.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    stem = Path(image_path).stem
    print(f"  [preprocess] Loaded {image.shape[1]}×{image.shape[0]}  ({image_path})")

    if save_debug:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: perspective correction ───────────────────────────────────────
    cropped, contour_skew = _detect_document(image)

    if cropped is None:
        warnings.warn(
            "Document boundary not detected — using full image. "
            "Check that the receipt is clearly visible against the background."
        )
        cropped      = image
        contour_skew = 0.0
    else:
        print(f"  [preprocess] After crop: {cropped.shape[1]}×{cropped.shape[0]}, "
              f"contour skew {contour_skew:.2f}°")

    if save_debug:
        cv2.imwrite(f"{debug_dir}/{stem}_1_cropped.jpg", cropped)

    # ── Step 2: coarse orientation (90° steps via Tesseract OSD) ─────────────
    coarse = _coarse_orientation(cropped)
    if coarse:
        print(f"  [preprocess] Coarse rotation applied: {coarse}°")
        cropped = _rotate_coarse(cropped, coarse)

        if save_debug:
            cv2.imwrite(f"{debug_dir}/{stem}_2_coarse_rotated.jpg", cropped)

    # ── Step 3: fine skew correction (sub-degree, from contour geometry) ─────
    # We use the angle derived from the contour in step 1, NOT from Hough lines
    # run on the cropped image — crumple lines and text dominate the Hough vote.
    if abs(contour_skew) >= 0.1:
        print(f"  [preprocess] Fine skew correction: {contour_skew:.2f}°")
        cropped = _rotate_fine(cropped, -contour_skew)

        if save_debug:
            cv2.imwrite(f"{debug_dir}/{stem}_3_deskewed.jpg", cropped)

    # ── Step 4: padding ───────────────────────────────────────────────────────
    result = _add_padding(cropped)
    print(f"  [preprocess] Final size: {result.shape[1]}×{result.shape[0]}")

    if save_debug:
        cv2.imwrite(f"{debug_dir}/{stem}_4_padded.jpg", result)
        print(f"  [preprocess] Debug images saved to '{debug_dir}/'")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Usage example
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from paddleocr import PaddleOCR

    IMAGE_PATH = "../testing-materials/receipt_deu_04.jpg"

    ocr = PaddleOCR(
        lang="de",
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
        # Disabled — handled by this script:
        use_doc_orientation_classify=True,

        use_doc_unwarping=True, 
        doc_unwarping_model_name="UVDoc",
        # Keep — useful for individual line orientation:
        use_textline_orientation=True,
        textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
        # Detector tuning for narrow receipt columns:
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.8,
    )

    preprocessed = preprocess_for_ocr(IMAGE_PATH, save_debug=True)
    result = ocr.predict(preprocessed)

    for res in result:
        res.print()
        res.save_to_img("output")  
        res.save_to_json("output")