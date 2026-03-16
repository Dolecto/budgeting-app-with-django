import os
import traceback
import sys
import mimetypes

import numpy as np
import pytesseract
import cv2

from pdf2image import convert_from_path
from scipy.ndimage import rank_filter


ACCEPTABLE_MIME = [
    "image/bmp",
    "image/png",
    "image/tiff",
    "image/jpeg",
    "image/jpg",
    "image/jpeg2000",
]


class FileNotAcceptedException(Exception):
    pass


class Page:
    def __init__(self, im, page_num, lang=None):
        self.healthy = True
        self.err = False
        self.page_num = page_num
        self.orig_im = im
        self.orig_shape = self.orig_im.shape
        self.lang = lang
        self.image = im

    def crop(self):
        try:
            self.image, self.num_tries = process_image(self.orig_im)
            return self.image
        except Exception as e:
            for frame in traceback.extract_tb(sys.exc_info()[2]):
                fname, lineno, _, _ = frame
                print(f"Error in {fname} on line {lineno}")
            print(e)
            self.err = e
            self.healthy = False

    def deskew(self):
        try:
            self.image, self.theta_est = process_skewed_crop(self.image)
            return self.image
        except Exception as e:
            self.err = e
            self.healthy = False

    def extract_text(self):
        self.text = pytesseract.image_to_string(self.image, lang=self.lang)
        return self.text
    

class Document:
    def __init__(self, lang=None):
        self.lang = lang
        self.pages = []
        self.processed_pages = []
        self.page_content = []
        self.prepared = False
        self.error = None

    def read(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self.file_extension = os.path.splitext(self.filename)[1].lower()

        mime_type, _ = mimetypes.guess_type(path)

        # --- PDF handling ---
        if mime_type == "application/pdf":
            images = convert_from_path(path, dpi=300)

            for i, pil_image in enumerate(images):
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
                page = Page(image, i, self.lang)
                self.pages.append(page)

            self.prepared = True

        # --- Image handling ---
        elif mime_type in ACCEPTABLE_MIME:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise Exception("Failed to load image.")
            page = Page(image, 0, self.lang)
            self.pages.append(page)
            self.prepared = True

        else:
            raise FileNotAcceptedException(
                "Supported formats: PDF, BMP, PNG, TIFF, JPG, JPEG2000"
            )

    def process(self):
        if not self.prepared:
            raise Exception("Call read() first.")

        for page in self.pages:
            page.crop()
            page.deskew()
            self.processed_pages.append(page)

    def extract_text(self):
        if not self.processed_pages:
            raise Exception("Run process() first.")

        for page in self.processed_pages:
            text = page.extract_text()
            self.page_content.append(text)

    def get_text(self):
        if not self.page_content:
            raise Exception("Run extract_text() first.")

        return "\n".join(self.page_content)
    


# ---------------- Image Processing ---------------- #

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def dilate(image, kernel, iterations):
    return cv2.dilate(image, kernel, iterations=iterations)


def downscale_image(im, max_dim=2048):
    h, w = im.shape[:2]
    if max(h, w) <= max_dim:
        return 1.0, im

    scale = max_dim / max(h, w)
    new_im = cv2.resize(im, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
    return scale, new_im


def find_components(im, max_components=16):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilation = dilate(im, kernel, 6)

    count = 21
    sigma = 0.0

    while count > max_components:
        sigma += 0.005
        contours, _ = cv2.findContours(
            dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        possible = find_likely_rectangles(contours, sigma)
        count = len(possible)

    return dilation, possible


def find_likely_rectangles(contours, sigma):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    possible = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, sigma * peri, True)
        box = make_box(approx)
        possible.append(box)

    return possible


def make_box(poly):
    xs = []
    ys = []
    for p in poly:
        for point in p:
            xs.append(point[0])
            ys.append(point[1])
    return min(xs), min(ys), max(xs), max(ys)


def rect_union(c1, c2):
    x11, y11, x21, y21 = c1
    x12, y12, x22, y22 = c2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def rect_area(c):
    x1, y1, x2, y2 = c
    return max(0, x2 - x1) * max(0, y2 - y1)


def crop_image(im, rect, scale):
    xmin, ymin, xmax, ymax = rect
    xmin, ymin, xmax, ymax = [int(x / scale) for x in rect]
    return im[ymin:ymax, xmin:xmax]


def reduce_noise_raw(im):
    bilat = cv2.bilateralFilter(im, 9, 75, 75)
    return cv2.medianBlur(bilat, 5)


def reduce_noise_edges(im):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, se)
    maxed_rows = rank_filter(opening, -4, size=(1, 20))
    maxed_cols = rank_filter(opening, -4, size=(20, 1))
    return np.minimum(np.minimum(opening, maxed_rows), maxed_cols)


def find_final_crop(rects):
    current = None
    for rect in rects:
        if current is None:
            current = rect
        else:
            current = rect_union(current, rect)
    return current


def process_image(orig_im):
    scale, im = downscale_image(orig_im)
    blur = reduce_noise_raw(im.copy())
    edges = auto_canny(blur.copy())
    debordered = reduce_noise_edges(edges.copy())
    dilation, rects = find_components(debordered)
    final_rect = find_final_crop(rects)
    cropped = crop_image(orig_im, final_rect, scale)
    return cropped, len(rects)


def estimate_skew(image):
    edges = auto_canny(image)
    lines = cv2.HoughLines(edges, 1, np.pi / 90, 200)

    if lines is None:
        return 0

    thetas = []

    for line in lines:
        for rho, theta in line:
            if np.pi / 3 < theta < np.pi * 2 / 3:
                thetas.append(theta)

    return np.degrees(np.mean(thetas)) if thetas else 0


def rotate(image, theta):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -theta, 1)
    return cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def process_skewed_crop(image):
    theta = estimate_skew(image)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rotated = rotate(thresh, theta)
    return rotated, theta