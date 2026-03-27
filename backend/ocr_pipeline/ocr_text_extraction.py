"""
ocr_pipeline/ocr_text_extraction.py
==================================
Extracts text from images given by ocr_preprocessing and outputs a structured JSON file
to be used in the frontend. 
"""

import os
import numpy as np
from paddleocr import PaddleOCR  


# User constants
# TODO: See what else to add here.
#     : Connect with user settings file.
USER_LANGUAGE = "de"


# Model constants
# https://www.paddleocr.ai/main/en/version3.x/module_usage/module_overview.html
TEXT_DET_MODEL = "PP-OCRv5_server_det"     
DOC_ORIENTATION_MODEL = "PP-LCNet_x1_0_doc_ori" 
DOC_UNWARPING_MODEL = "UVDoc"
TEXTLINE_MODEL = "PP-LCNet_x1_0_textline_ori"
PADDLE_VERSION = "PP-OCRv5"


# PaddleOCR-specific flags
# TODO: Go through https://github.com/PaddlePaddle/PaddleOCR and find these flags for faster inference.
#     : Actually go through these flags and see if they break anything.
# os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = True
# os.environ["PADDLE_PDX_CACHE_HOME"] = False


# https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html#2-list-of-supported-models
# TODO: List additional languages that are in the same script family.
#     : Check docs for their locale code standard.
#     : See if we need to map country codes to locale code.
# This is inferred by `lang` in PaddleOCR, but we'll probably need to specify the model to ensure fast models anyway.
# Just need to improve this by including a country-locale mapper and feed that to PaddleOCR.
text_rec_model_map = {
    # Latin 
    "en": "en_PP-OCRv5_mobile_rec",
    "de": "latin_PP-OCRv5_mobile_rec",

    # Cyrillic
    # TODO: Verify.
    "ru": "cyrillic_PP-OCRv5_mobile_rec",
    "uk": "eslav_PP-OCRv5_mobile_rec",       # Ukrainian 

    # East Asia 
    "zh": "PP-OCRv5_server_rec",             # simplified Chinese
    "ko": "korean_PP-OCRv5_mobile_rec",

    # Middle East
    "ar": "arabic_PP-OCRv5_mobile_rec",      # unsure if Modern Standard Arabic

    # Indian subcontinent
    "hi": "devanagari_PP-OCRv5_mobile_rec",  # Hindi
    "te": "te_PP-OCRv5_mobile_rec",          # Telugu
    "ta": "ta_PP-OCRv5_mobile_rec",          # Tamil

    # Special
    "th": "th_PP-OCRv5_mobile_rec",          # Thai
    "el": "el_PP-OCRv5_mobile_rec",          # Greek
}


# https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_pipelines/ocr.py
# TODO: Figure out how to keep model loaded at start up (unless settings change).
def extract_text(img: np.ndarray, json_output_dir: str = "json_outputs", debug: bool = False, debug_dir: str = "ocr_debugging"):
    ocr = PaddleOCR(
        lang=USER_LANGUAGE,
        ocr_version=PADDLE_VERSION,
        text_detection_model_name=TEXT_DET_MODEL, 
        text_recognition_model_name=text_rec_model_map[USER_LANGUAGE],

        use_doc_orientation_classify=False,     
        doc_orientation_classify_model_name=DOC_ORIENTATION_MODEL,

        use_doc_unwarping=True, 
        doc_unwarping_model_name=DOC_UNWARPING_MODEL,

        use_textline_orientation=True, 
        textline_orientation_model_name=TEXTLINE_MODEL,

        text_det_unclip_ratio=1.2,
    )

    result = ocr.predict(img)  

    os.makedirs(json_output_dir, exist_ok=True)

    # Loop over pages
    for res in result:  
        # res.print()

        res.save_to_json(json_output_dir)
        
        if debug:
            os.makedirs(debug_dir, exist_ok=True)
            res.save_to_img(debug_dir)  

    num_pages = len(result)
    print(f"Saved {num_pages} page/s in {json_output_dir}")
    if debug: print(f"Saved {num_pages} image/s in {debug_dir}")


