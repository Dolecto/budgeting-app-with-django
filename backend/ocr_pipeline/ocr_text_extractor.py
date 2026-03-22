"""
ocr_pipeline/ocr_text_extractor.py
==================================
Prepares tiles from ocr_tiler to be used in ocr_text_extractor.


"""
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


def extract_text(img):
    custom_config = f"--oem 3 --psm 4 -l deu --dpi 300"  # -c tessedit_char_whitelist={whitelist} -psm 4
    raw_text = pytesseract.image_to_string(img, config=custom_config) 

    raw_text = pytesseract.image_to_string(img, config=custom_config, 
                                            output_type=pytesseract.Output.STRING)
    return raw_text


def preprocess_receipt(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    # enhance_for_ocr already handled contrast/sharpening
    w, h = image.size
    if w < 1000:
        image = image.resize((w * 2, h * 2), Image.LANCZOS)
    return image