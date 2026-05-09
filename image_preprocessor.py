"""
Image Preprocessor Module
Handles image cleaning and preparation for OCR processing.
Includes grayscale conversion, denoising, thresholding, and deskewing.

Phase 2: Added adaptive preprocessing with configurable strategies.
"""

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Main preprocessing pipeline (Phase 1 — legacy fallback).
    Takes a PIL Image and returns a cleaned PIL Image ready for OCR.
    """
    # Convert to RGB if needed (handles RGBA, L, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Adaptive thresholding for binarization
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )

    # Deskew
    deskewed = deskew_image(binary)

    return Image.fromarray(deskewed)


def preprocess_adaptive(image: Image.Image, strategy) -> Image.Image:
    """
    Adaptive preprocessing pipeline (Phase 2 — agentic).
    Applies preprocessing based on the agent's selected strategy.
    
    Args:
        image: PIL Image to preprocess
        strategy: PreprocessingStrategy object with parameters
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Step 1: Denoising (if enabled)
    if strategy.do_denoise:
        h_param = strategy.denoise_strength
        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=h_param, templateWindowSize=7, searchWindowSize=21
        )
    else:
        denoised = gray

    # Step 2: Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=strategy.clahe_clip,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(denoised)

    # Step 3: Sharpening (if enabled)
    if strategy.do_sharpen:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Step 4: Adaptive thresholding (if enabled)
    if strategy.do_threshold:
        result = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            strategy.adaptive_block_size,
            strategy.adaptive_c
        )
    else:
        result = enhanced

    # Step 5: Deskew (if enabled)
    if strategy.do_deskew:
        result = deskew_image(result)

    return Image.fromarray(result)


def analyze_image_quality(image: Image.Image) -> dict:
    """
    Analyze image quality metrics.
    Returns a dictionary with blur, noise, contrast, and brightness scores.
    Used by the agent for perception.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Blur detection (Laplacian variance — higher = sharper)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Noise estimation
    noise_level = float(np.std(cv2.Laplacian(gray, cv2.CV_64F)))

    # Contrast ratio
    contrast = float(gray.max() - gray.min()) / 255.0

    # Brightness
    brightness = float(np.mean(gray)) / 255.0

    return {
        'blur_score': round(blur_score, 2),
        'noise_level': round(noise_level, 2),
        'contrast': round(contrast, 3),
        'brightness': round(brightness, 3),
        'resolution': image.size,
    }


def get_color_mask(image: Image.Image) -> np.ndarray:
    """
    Extract red-colored regions from the image.
    Used for detecting headings/emphasized text written in red ink.
    Returns a binary mask where red regions are white.
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_array = np.array(image)

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Red color range in HSV (red wraps around 0/180)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    return red_mask


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Correct image skew using minAreaRect detection.
    """
    # Find contours
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 10:
        return image

    # Get the angle of the minimum area bounding rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only deskew if angle is significant but not too large
    if abs(angle) < 0.5 or abs(angle) > 15:
        return image

    # Rotate
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def resize_for_ocr(image: Image.Image, max_dimension: int = 3000) -> Image.Image:
    """
    Resize image if too large, maintaining aspect ratio.
    Tesseract works best with images around 300 DPI.
    """
    w, h = image.size
    if max(w, h) <= max_dimension:
        return image

    scale = max_dimension / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return image.resize((new_w, new_h), Image.LANCZOS)
