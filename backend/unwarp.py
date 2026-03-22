import json
from pathlib import Path

from PIL import Image

import pytesseract
from PIL import ImageFilter, ImageEnhance


import os
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


class Unwarp:
    def __init__(
        self,
        model_path: str = os.path.join(
            os.path.dirname(__file__), "models", "uvdoc.onnx"
        ),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
        image_size: Tuple[int, int] = (488, 712),
        grid_size: Tuple[int, int] = (45, 31),
    ):
        self.image_size = image_size
        self.grid_size = grid_size
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        self.bilinear_unwarping = ort.InferenceSession(
            os.path.join(
                os.path.dirname(__file__), "models", "bilinear_unwarping.onnx"
            )
        )

    def prepare_input(
        self, image: Image.Image
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        original_size = image.size
        image_array = np.array(image)

        resized_image = image.resize(self.image_size)
        resized_array = np.array(resized_image)

        normalized_original = image_array.transpose(2, 0, 1) / 255
        normalized_resized = resized_array.transpose(2, 0, 1) / 255

        return (
            np.expand_dims(normalized_resized, 0),
            np.expand_dims(normalized_original, 0),
            original_size,
        )

    def inference(self, image: Image.Image) -> Image.Image:
        resized_input, original_input, original_size = self.prepare_input(image)

        points, _ = self.session.run(None, {"input": resized_input.astype(np.float16)})

        unwarped = self.bilinear_unwarping.run(
            None,
            {
                "warped_img": original_input.astype(np.float32),
                "point_positions": points.astype(np.float32),
                "img_size": np.array(original_size),
            },
        )[0][0]

        unwarped_array = (unwarped.transpose(1, 2, 0) * 255).astype(np.uint8)

        return Image.fromarray(unwarped_array)


def preprocess_receipt(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    w, h = image.size
    if w < 1000:
        image = image.resize((w * 2, h * 2), Image.LANCZOS)
    return image


def ocr_receipt(image_path: str) -> dict:
    unwarper = Unwarp()

    # Step 1: Load & dewarp
    print(f"Loading image: {image_path}")
    raw = Image.open(image_path).convert("RGB")
    print(f"Original size: {raw.size}")

    print("Dewarping...")
    dewarped = unwarper.inference(raw)
    print(f"Dewarped size: {dewarped.size}")

    # Step 2: Preprocess for OCR
    print("Preprocessing for OCR...")
    processed = preprocess_receipt(dewarped)

    # Step 3: OCR
    print("Running OCR...")
    whitelist = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüÄÖÜß .,$/€£%:-@#&()"
    custom_config = f"--oem 3 --psm 6 -l deu --dpi 300"  # -c tessedit_char_whitelist={whitelist} -psm 4
    raw_text = pytesseract.image_to_string(processed, config=custom_config) 

    # Step 4: Also grab per-word confidence scores
    data = pytesseract.image_to_data(
        processed, config=custom_config, output_type=pytesseract.Output.DICT
    )
    words = [
        {"word": w, "confidence": c}
        for w, c in zip(data["text"], data["conf"])
        if w.strip() and int(c) > 40  # filter low-confidence and empty tokens
    ]

    return {
        "raw_text": raw_text.strip(),
        "words": words,
        "avg_confidence": round(
            sum(w["confidence"] for w in words) / len(words), 2
        ) if words else 0,
    }


def save_intermediates(image_path: str, output_dir: str = "output") -> None:
    """Save dewarped and preprocessed images for debugging."""
    Path(output_dir).mkdir(exist_ok=True)
    stem = Path(image_path).stem

    unwarper = Unwarp()
    raw = Image.open(image_path).convert("RGB")

    dewarped = unwarper.inference(raw)
    dewarped.save(f"{output_dir}/{stem}_dewarped.png")
    print(f"Saved dewarped image to {output_dir}/{stem}_dewarped.png")

    processed = preprocess_receipt(dewarped)
    processed.save(f"{output_dir}/{stem}_preprocessed.png")
    print(f"Saved preprocessed image to {output_dir}/{stem}_preprocessed.png")


if __name__ == "__main__":
    IMAGE_PATH = "../testing-materials/receipt_deu_01.jpg"  # <- replace with your image

    # Save intermediate images so you can verify dewarping visually
    save_intermediates(IMAGE_PATH)

    # Run full OCR pipeline
    result = ocr_receipt(IMAGE_PATH)

    print("\n--- RAW TEXT ---")
    print(result["raw_text"])

    print(f"\n--- STATS ---")
    print(f"Average OCR confidence: {result['avg_confidence']}%")
    print(f"Words detected: {len(result['words'])}")

    print("\n--- LOW CONFIDENCE WORDS (< 70%) ---")
    low_conf = [w for w in result["words"] if w["confidence"] < 70]
    for item in low_conf:
        print(f"  '{item['word']}' — {item['confidence']}%")

    # Optionally dump full result to JSON
    with open("output/result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFull result saved to output/result.json")
