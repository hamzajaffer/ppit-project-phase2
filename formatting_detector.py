"""
Formatting Detector Module
Detects text formatting attributes (headings, bold, alignment, paragraphs)
from visual cues in the original image and OCR positional data.

Phase 2: Added adaptive formatting, bullet/list detection.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from image_preprocessor import get_color_mask


def detect_formatting(
    original_image: Image.Image,
    text_blocks: List[Dict[str, Any]],
    image_width: int
) -> List[Dict[str, Any]]:
    """Analyze each text block and assign formatting attributes."""
    red_mask = get_color_mask(original_image)

    for block in text_blocks:
        block['is_heading'] = is_heading(block, red_mask)
        block['is_bold'] = is_bold_text(block, red_mask)
        block['alignment'] = detect_alignment(block, image_width)
        block['font_size_category'] = estimate_font_size(block, text_blocks)
        block['is_bullet'] = detect_bullet(block)

    text_blocks = detect_paragraph_breaks(text_blocks)
    return text_blocks


def is_heading(block, red_mask, threshold=0.15):
    """Detect if a text block is a heading via red color or markers."""
    top = max(0, int(block['top'] - 5))
    bottom = min(red_mask.shape[0], int(block['top'] + block['height'] + 5))
    left = max(0, int(block['left'] - 5))
    if block.get('words'):
        last_word = block['words'][-1]
        right = min(red_mask.shape[1], last_word['left'] + last_word['width'] + 5)
    else:
        right = min(red_mask.shape[1], left + 50 * block['word_count'])
    if top >= bottom or left >= right:
        return False
    region = red_mask[top:bottom, left:right]
    if region.size == 0:
        return False
    red_ratio = np.count_nonzero(region) / region.size
    if red_ratio > threshold:
        return True
    text = block['text'].strip()
    if text.startswith('#') or text.startswith('*'):
        return True
    return False


def is_bold_text(block, red_mask, threshold=0.10):
    """Detect if text appears bold via red color."""
    top = max(0, int(block['top'] - 3))
    bottom = min(red_mask.shape[0], int(block['top'] + block['height'] + 3))
    left = max(0, int(block['left'] - 3))
    if block.get('words'):
        last_word = block['words'][-1]
        right = min(red_mask.shape[1], last_word['left'] + last_word['width'] + 3)
    else:
        right = min(red_mask.shape[1], left + 200)
    if top >= bottom or left >= right:
        return False
    region = red_mask[top:bottom, left:right]
    if region.size == 0:
        return False
    red_ratio = np.count_nonzero(region) / region.size
    return red_ratio > threshold


def detect_alignment(block, image_width):
    """Detect text alignment based on x-position."""
    left_margin = block['left']
    if left_margin < image_width * 0.3:
        return 'left'
    elif left_margin > image_width * 0.6:
        return 'right'
    return 'center'


def detect_bullet(block):
    """Phase 2: Detect if a text block is a bullet point or list item."""
    text = block['text'].strip()
    bullets = ['•', '●', '○', '■', '□', '▪', '-', '–', '—']
    for b in bullets:
        if text.startswith(b):
            return True
    if len(text) > 2 and text[0].isdigit():
        dot_pos = text.find('.')
        if 0 < dot_pos <= 3 and dot_pos < len(text) - 1:
            return True
    if len(text) > 2 and text[0].isalpha() and text[1] == '.' and text[2] == ' ':
        return True
    return False


def estimate_font_size(block, all_blocks):
    """Estimate relative font size based on text height."""
    if not all_blocks:
        return 'normal'
    heights = [b['height'] for b in all_blocks if b['height'] > 0]
    if not heights:
        return 'normal'
    median_height = sorted(heights)[len(heights) // 2]
    if block['height'] > median_height * 1.4:
        return 'large'
    elif block['height'] < median_height * 0.7:
        return 'small'
    return 'normal'


def detect_paragraph_breaks(blocks):
    """Detect paragraph breaks based on vertical spacing."""
    if len(blocks) < 2:
        for block in blocks:
            block['new_paragraph'] = True
        return blocks
    gaps = []
    for i in range(1, len(blocks)):
        gap = blocks[i]['top'] - (blocks[i-1]['top'] + blocks[i-1]['height'])
        gaps.append(max(0, gap))
    if not gaps:
        for block in blocks:
            block['new_paragraph'] = True
        return blocks
    avg_gap = sum(gaps) / len(gaps)
    paragraph_threshold = avg_gap * 1.8 if avg_gap > 0 else 20
    blocks[0]['new_paragraph'] = True
    for i in range(1, len(blocks)):
        gap = blocks[i]['top'] - (blocks[i-1]['top'] + blocks[i-1]['height'])
        blocks[i]['new_paragraph'] = gap > paragraph_threshold
    return blocks
