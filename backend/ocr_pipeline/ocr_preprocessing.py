"""
ocr_pipeline/ocr_preprocessing.py
====================
A collection of preprocessing functions for images to be fed into ocr_text_extraction.

Includes a OCRPreprocessingPipeline class to allow creation of multiple pipelines for confidence voting.

Pipelines so far:
1. upscale(1.15) - denoise("bilateral") - enhance("clahe+bilateral+unsharp")   # for some reason, these are different
2. normalize - denoise("bilateral") - enhance("clahe") - upscale(1.15)         # 
"""

import numpy as np
import cv2


class OCRPreprocessingPipeline:
    """
    A sequential image preprocessing pipeline for OCR.
    
    Parameters
    ----------
    steps : list[callable]
          A list of preprocessing functions. Function parameters should be specified here.
    name : str
        A name for this pipeline for debugging purposes.

    Usage
    -----
    pipeline = OCRPreprocessingPipeline(steps=[
        normalize,
        lambda img: denoise(img, method="bilateral"),
        lambda img: enhance(img, method="clahe"),
        lambda img: upscale(img, scale=1.15)
        ])

    result = pipeline.run(image)
    """
    def __init__(self, steps: list[callable], name: str = ""):
        self.steps = steps
        self.name = name    


    def load(self, path: str) -> np.ndarray:
        """Load an image from a file path into a BGR uint8 numpy array."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img


    def run(self, image_path: str):
        """Run the preprocessing functions in sequence."""
        # TODO: figure out how to run this in parallel
        img = self.load(image_path)

        for step in self.steps:
            img = step(img)

        return img


    # Helper functions
    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return img.astype(np.uint8)
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


    # Preprocessing functions
    def upscale(
        self,
        image: np.ndarray,
        method: str = "lanczos",
        min_width: int = 1240,
        scale: float | None = None,
        max_side: int = 4000,
    ) -> np.ndarray:
        """
        Resize the image to improve detection of small / dense text.

        Parameters
        ----------
        method : str
            Interpolation algorithm:
                - 'lanczos' 
                - 'cubic'   
                - 'linear'  
        min_width : int
            Upscale until the image is at least this many pixels wide.
            Ignored when `scale` is provided. 
        scale : float | None
            Explicit scale factor (e.g. 2.0 doubles both dimensions).
            When set, `min_width` is ignored.
        max_side : int
            Hard cap on the longer dimension to avoid memory issues.
            Default 4000.

        Returns
        -------
        np.ndarray  BGR uint8
        """
        interp = {
            "lanczos": cv2.INTER_LANCZOS4,
            "cubic":   cv2.INTER_CUBIC,
            "linear":  cv2.INTER_LINEAR,
        }
        if method not in interp:
            raise ValueError(f"upscale: method must be one of {list(interp)}, got '{method}'")

        h, w = image.shape[:2]

        if scale is not None:
            fx = fy = scale
        else:
            fx = fy = max(1.0, min_width / w)

        new_w = int(w * fx)
        new_h = int(h * fy)

        # Apply max_side cap
        longest = max(new_w, new_h)
        if longest > max_side:
            cap   = max_side / longest
            new_w = int(new_w * cap)
            new_h = int(new_h * cap)

        if new_w == w and new_h == h:
            return image

        return cv2.resize(image, (new_w, new_h), interpolation=interp[method])


    def enhance(
        self,
        image: np.ndarray,
        method: str = "clahe+bilateral+unsharp",
    ) -> np.ndarray:
        """
        Contrast and clarity enhancement.

        Parameters
        ----------
        method : str
            Enhancement recipe:
            - 'clahe+bilateral+unsharp'
                CLAHE on the L channel → bilateral smoothing → unsharp mask.
            - 'clahe'
                Good for well-lit images.
            - 'bg_subtract+clahe'
                Divide by blurred background first (remove uneven lighting),
                then CLAHE.  Best for strongly shadowed or crumpled paper.
            - 'gamma'
                Simple gamma correction to lift dark images.

        Returns
        -------
        np.ndarray  BGR uint8
        """
        if method == "clahe+bilateral+unsharp":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
            blur  = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
            image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)

        elif method == "clahe":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        elif method == "bg_subtract+clahe":
            gray = self._to_gray(image).astype(np.float32)
            bg   = cv2.GaussianBlur(gray, (91, 91), 0)
            bg   = np.where(bg == 0, 1, bg)
            norm = (gray / bg * 200).clip(0, 255).astype(np.uint8)
            image = self._to_bgr(norm)
            lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        elif method == "gamma":
            lut   = (np.arange(256) / 255.0) ** (1.0 / 1.5) * 255
            image = lut[image].astype(np.uint8)

        else:
            raise ValueError(
                f"enhance: method must be one of "
                f"['clahe+bilateral+unsharp', 'clahe', 'bg_subtract+clahe', 'gamma'], "
                f"got '{method}'"
            )

        return image


    def denoise(
        self,
        image: np.ndarray,
        method: str = "nlm",
        h: float = 10.0,
    ) -> np.ndarray:
        """
        Noise reduction.

        Parameters
        ----------
        method : str
            Denoising algorithm:
            - 'nlm'       - Non-local means.  Best quality, slower.
                            Handles grainy phone photos well.
            - 'gaussian'  - Gaussian blur.  Softens edges.
            - 'median'    - Median filter.  Good for salt-and-pepper noise
                            (scanning artifacts, dust).
            - 'bilateral' - Edge-preserving smoothing.  Slower than Gaussian
                            but keeps text strokes sharp.
        h : float
            Filter strength for 'nlm'.  Higher = smoother, but may blur thin
            strokes.  Typical range 5-15.

        Returns
        -------
        np.ndarray  BGR uint8
        """
        if method == "nlm":
            if image.ndim == 2:
                image = cv2.fastNlMeansDenoising(image, None, h, 7, 21)
            else:
                image = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

        elif method == "gaussian":
            image = cv2.GaussianBlur(image, (3, 3), 0)

        elif method == "median":
            image = cv2.medianBlur(image, 3)

        elif method == "bilateral":
            image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        else:
            raise ValueError(
                f"denoise: method must be one of "
                f"['nlm', 'gaussian', 'median', 'bilateral'], got '{method}'"
            )

        return self._to_bgr(image)


    def sharpen(
        self,
        image: np.ndarray,
        method: str = "unsharp",
        strength: float = 1.5,
    ) -> np.ndarray:
        """
        Edge / stroke sharpening.

        Parameters
        ----------
        method : str
            Sharpening algorithm:
            - 'unsharp'   - Unsharp masking.
            - 'laplacian' - Laplacian kernel.  Can amplify noise
                            if the image isn't already clean.
            - 'highboost' - High-boost filter via Gaussian subtraction.
                            Between unsharp and laplacian in aggressiveness.
        strength : float
            Sharpening intensity.  1.0 = moderate, 2.0 = aggressive.

        Returns
        -------
        np.ndarray  BGR uint8
        """
        if method == "unsharp":
            blur  = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
            image = cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

        elif method == "laplacian":
            kernel = np.array([
                [ 0, -1,  0],
                [-1,  5, -1],
                [ 0, -1,  0],
            ], dtype=np.float32)
            image = cv2.filter2D(image, -1, kernel * strength)
            image = np.clip(image, 0, 255).astype(np.uint8)

        elif method == "highboost":
            blur  = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
            mask  = cv2.subtract(image, blur)
            image = cv2.addWeighted(image, 1.0, mask, strength, 0)

        else:
            raise ValueError(
                f"sharpen: method must be one of "
                f"['unsharp', 'laplacian', 'highboost'], got '{method}'"
            )

        return self._to_bgr(image)


    def binarize(
        self,
        image: np.ndarray,
        method: str = "adaptive",
        block_size: int = 31,
        C: int = 10,
    ) -> np.ndarray:
        """
        Convert to black-and-white.

        Parameters
        ----------
        method : str
            Binarization algorithm:
            - 'adaptive'  - Adaptive Gaussian thresholding.
                            Best for uneven backgrounds.
            - 'otsu'      - Global Otsu threshold.  Works well after
                            illumination normalization (enhance / bg_subtract).
        block_size : int
            Neighbourhood size for 'adaptive' (must be odd).
        C : int
            Constant subtracted from the local mean for 'adaptive'. 

        Returns
        -------
        np.ndarray  BGR uint8  (pixel values are 0 or 255)
        """
        gray = self._to_gray(image)

        if method == "adaptive":
            if block_size % 2 == 0:
                block_size += 1
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size, C,
            )

        elif method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        else:
            raise ValueError(
                f"binarize: method must be one of "
                f"['adaptive', 'otsu'], got '{method}'"
            )

        return self._to_bgr(binary)


    def deskew(
        self,
        image: np.ndarray,
        method: str = "minarea",
        max_angle: float = 10.0,
        border_value: int = 255,
    ) -> np.ndarray:
        """
        Straighten a rotated or slightly tilted document.

        Parameters
        ----------
        method : str
            Detection algorithm:
            - 'minarea'  - Minimum-area bounding rectangle of all
                            foreground pixels. 
            - 'hough'    - Hough line transform. 
        max_angle : float
            Maximum rotation angle in degrees to correct.  Skips correction if
            the detected angle exceeds this to avoid large mis-rotations on
            noisy images.
        border_value : int
            Fill colour for areas exposed after rotation.  255 = white.

        Returns
        -------
        np.ndarray  BGR uint8
        """
        gray = self._to_gray(image)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        if method == "minarea":
            coords = np.column_stack(np.where(binary > 0))
            if coords.size == 0:
                return image
            rect  = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle += 90

        elif method == "hough":
            edges  = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            if lines is None:
                return image
            angles = [
                np.degrees(line[0][1]) - 90
                for line in lines
                if abs(np.degrees(line[0][1]) - 90) < max_angle
            ]
            if not angles:
                return image
            angle = float(np.median(angles))

        else:
            raise ValueError(
                f"deskew: method must be one of ['minarea', 'hough'], got '{method}'"
            )

        if abs(angle) > max_angle:
            print(f"deskew: detected angle {angle:.1f}° exceeds max_angle\n")
            print(f"({max_angle}°) - skipping.") 
         
            return image

        h, w    = image.shape[:2]
        M       = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(border_value,) * 3,
        )
        return rotated


    def normalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize pixel intensity values.

        Returns
        -------
        np.ndarray  BGR uint8
        """
        gray = self._to_gray(image).astype(np.float32)

        lo, hi = gray.min(), gray.max()
        if hi == lo:
            return image   # flat image, nothing to normalize
        normalized = (gray - lo) / (hi - lo) * 255

        return self._to_bgr(normalized.clip(0, 255).astype(np.uint8))