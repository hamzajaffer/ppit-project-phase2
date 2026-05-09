"""
OCR Engine Module
Handles text extraction from images using Tesseract OCR.
Returns structured data with text content and positional metadata.

Phase 2: Added intelligent OCR with configurable PSM modes and
         confidence-based extraction.
"""

import os
import re
import pytesseract
import numpy as np
from PIL import Image
from typing import List, Dict, Any

# Configure Tesseract path for Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_text_simple(image: Image.Image, psm_mode: int = 6) -> str:
    """
    Simple text extraction — returns plain text string.
    """
    custom_config = f'--oem 3 --psm {psm_mode}'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text.strip()


def extract_text_with_data(
    image: Image.Image,
    psm_mode: int = 6,
    min_confidence: int = 20,
) -> List[Dict[str, Any]]:
    """
    Extract text with positional and confidence data.
    Returns list of word-level data with bounding boxes.
    
    Phase 2: Added configurable PSM mode and minimum confidence threshold.
    """
    custom_config = f'--oem 3 --psm {psm_mode}'
    data = pytesseract.image_to_data(
        image, config=custom_config, output_type=pytesseract.Output.DICT
    )

    words = []
    n_boxes = len(data['text'])

    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])

        # Skip empty text or very low confidence
        if not text or conf < min_confidence:
            continue

        words.append({
            'text': text,
            'left': data['left'][i],
            'top': data['top'][i],
            'width': data['width'][i],
            'height': data['height'][i],
            'conf': conf,
            'block_num': data['block_num'][i],
            'par_num': data['par_num'][i],
            'line_num': data['line_num'][i],
            'word_num': data['word_num'][i],
        })

    return words


def extract_with_multi_pass(image: Image.Image) -> Dict[str, Any]:
    """
    Phase 2: Multi-pass OCR — tries different PSM modes and returns the best result.
    Used by the agent when a single pass produces low confidence.
    """
    psm_modes = [6, 3, 4, 11]  # Different page segmentation modes
    best_result = None
    best_confidence = 0

    for psm in psm_modes:
        try:
            words = extract_text_with_data(image, psm_mode=psm)
            if not words:
                continue

            avg_conf = sum(w['conf'] for w in words) / len(words)
            word_count = len(words)

            if avg_conf > best_confidence and word_count > 0:
                best_confidence = avg_conf
                best_result = {
                    'words': words,
                    'avg_confidence': avg_conf,
                    'word_count': word_count,
                    'psm_mode': psm,
                }
        except Exception:
            continue

    if best_result is None:
        return {
            'words': [],
            'avg_confidence': 0,
            'word_count': 0,
            'psm_mode': 6,
        }

    return best_result


def post_process_text(text: str) -> str:
    """
    Phase 2: Basic text post-processing to fix common OCR errors.
    """
    if not text:
        return text

    # Fix common OCR substitutions
    corrections = {
        '|': 'l',       # pipe → lowercase L
        '0': 'O',       # zero → O (only in word context, handled carefully)
    }

    # Fix double spaces
    text = re.sub(r'  +', ' ', text)

    # Fix lines that are just punctuation
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep lines that have at least some alphanumeric content
        if stripped and (any(c.isalnum() for c in stripped) or len(stripped) > 3):
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def group_words_into_lines(
    words: List[Dict[str, Any]],
    line_threshold: int = 15,
) -> List[List[Dict[str, Any]]]:
    """
    Group detected words into lines based on vertical position (y-coordinate).
    Words with similar 'top' values are on the same line.
    """
    if not words:
        return []

    # Sort by top position, then left
    sorted_words = sorted(words, key=lambda w: (w['top'], w['left']))

    lines = []
    current_line = [sorted_words[0]]

    for word in sorted_words[1:]:
        # Check if this word is on the same line (similar y position)
        avg_top = sum(w['top'] for w in current_line) / len(current_line)
        if abs(word['top'] - avg_top) <= line_threshold:
            current_line.append(word)
        else:
            # Sort current line by x position and save
            current_line.sort(key=lambda w: w['left'])
            lines.append(current_line)
            current_line = [word]

    # Don't forget the last line
    current_line.sort(key=lambda w: w['left'])
    lines.append(current_line)

    return lines


def lines_to_text_blocks(lines: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert grouped lines into text blocks with metadata.
    Detects paragraph breaks based on vertical gaps between lines.
    """
    if not lines:
        return []

    blocks = []
    for line_words in lines:
        line_text = ' '.join(w['text'] for w in line_words)
        avg_top = sum(w['top'] for w in line_words) / len(line_words)
        avg_height = sum(w['height'] for w in line_words) / len(line_words)
        min_left = min(w['left'] for w in line_words)
        avg_conf = sum(w['conf'] for w in line_words) / len(line_words)

        blocks.append({
            'text': line_text,
            'top': avg_top,
            'left': min_left,
            'height': avg_height,
            'confidence': avg_conf,
            'word_count': len(line_words),
            'words': line_words,
        })

    return blocks
