import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import os

import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from ocr_pipeline.ocr_tiler import get_tiles, enhance_for_ocr
from ocr_pipeline.ocr_dewarper import Dewarp
from ocr_pipeline.ocr_text_extractor import extract_text, preprocess_receipt



TEST_IMAGE = "../testing-materials/receipt_deu_03.jpg"



image = cv2.imread(TEST_IMAGE)
dewarper = Dewarp()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
pil = Image.fromarray(rgb)
dewarped_full = dewarper.inference(pil)

dewarped_full_bgr = cv2.cvtColor(np.array(dewarped_full), cv2.COLOR_RGB2BGR)
enhanced_bgr = enhance_for_ocr(dewarped_full_bgr)
enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
processed_full = preprocess_receipt(enhanced_pil)

text_full = extract_text(processed_full)
print(text_full)
print("--------------------------------------------\n")

"""
with open("demofile.html", "a") as f:
    f.write(f"<pre>{text_full}</pre><hr>")
    f.write("--------------------------------------------\n")
"""
os.makedirs("dewarper", exist_ok=True)
dewarped_full.save("dewarper/dewarped_full.png")

i = 0
for tile in get_tiles(image):
    rgb = cv2.cvtColor(tile["image"], cv2.COLOR_BGR2RGB).copy()
    pil_image = Image.fromarray(rgb)
    dewarped = dewarper.inference(pil_image)

    dewarped_bgr = cv2.cvtColor(np.array(dewarped), cv2.COLOR_RGB2BGR)
    enhanced = Image.fromarray(enhance_for_ocr(dewarped_bgr))

    preprocessed = preprocess_receipt(enhanced)

    os.makedirs("dewarper", exist_ok=True)
    dewarped.save(f"dewarper/dewarped0{i}.png")
    i += 1
    text = extract_text(preprocessed)
    print(text)
    print("--------------------------------------------\n")

    """
    with open("demofile.html", "a") as f:
        f.write(f"<pre>{text}</pre><hr>")
        f.write("--------------------------------------------\n")
    """



