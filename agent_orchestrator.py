"""
Agent Orchestrator Module
The core agentic brain of the OCR Image-to-Word Converter.
Implements the Perceive → Decide → Act → Learn cycle.
Manages adaptive preprocessing, intelligent OCR, and quality gates.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from memory_system import AgentMemory, ConversionRecord
from safety_logger import SafetyLogger


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ImageAnalysis:
    """Results of the agent's perception of an image."""
    blur_score: float = 0.0
    noise_level: float = 0.0
    contrast_ratio: float = 0.0
    brightness: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    has_color: bool = False
    estimated_text_density: float = 0.0
    quality_profile: str = "unknown"  # 'clean', 'noisy', 'blurry', 'low_contrast', 'mixed'
    quality_score: float = 0.0  # 0-100

    def to_dict(self) -> dict:
        return {
            'blur_score': round(self.blur_score, 2),
            'noise_level': round(self.noise_level, 2),
            'contrast_ratio': round(self.contrast_ratio, 2),
            'brightness': round(self.brightness, 2),
            'resolution': f"{self.resolution[0]}×{self.resolution[1]}",
            'has_color': self.has_color,
            'text_density': round(self.estimated_text_density, 2),
            'quality_profile': self.quality_profile,
            'quality_score': round(self.quality_score, 1),
        }


@dataclass
class PreprocessingStrategy:
    """A preprocessing strategy selected by the agent."""
    name: str
    description: str
    denoise_strength: int = 15
    clahe_clip: float = 2.0
    adaptive_block_size: int = 15
    adaptive_c: int = 8
    do_deskew: bool = True
    do_denoise: bool = True
    do_threshold: bool = True
    do_sharpen: bool = False
    psm_mode: int = 6  # Tesseract page segmentation mode

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'denoise_strength': self.denoise_strength,
            'clahe_clip': self.clahe_clip,
            'block_size': self.adaptive_block_size,
            'adaptive_c': self.adaptive_c,
            'deskew': self.do_deskew,
            'denoise': self.do_denoise,
            'threshold': self.do_threshold,
            'sharpen': self.do_sharpen,
            'psm_mode': self.psm_mode,
        }


# ─── Predefined Strategies ───────────────────────────────────────────────────

STRATEGIES = {
    'balanced': PreprocessingStrategy(
        name='balanced',
        description='Standard preprocessing for typical documents',
        denoise_strength=15,
        clahe_clip=2.0,
        adaptive_block_size=15,
        adaptive_c=8,
        psm_mode=6,
    ),
    'aggressive': PreprocessingStrategy(
        name='aggressive',
        description='Heavy denoising and contrast enhancement for noisy/poor images',
        denoise_strength=25,
        clahe_clip=3.5,
        adaptive_block_size=21,
        adaptive_c=12,
        do_sharpen=True,
        psm_mode=6,
    ),
    'light': PreprocessingStrategy(
        name='light',
        description='Minimal preprocessing for clean, high-quality images',
        denoise_strength=5,
        clahe_clip=1.5,
        adaptive_block_size=11,
        adaptive_c=5,
        do_deskew=False,
        psm_mode=6,
    ),
    'handwritten': PreprocessingStrategy(
        name='handwritten',
        description='Optimized for handwritten text with adaptive parameters',
        denoise_strength=10,
        clahe_clip=2.5,
        adaptive_block_size=19,
        adaptive_c=10,
        do_sharpen=True,
        psm_mode=6,
    ),
    'high_contrast': PreprocessingStrategy(
        name='high_contrast',
        description='Maximum contrast enhancement for low-contrast images',
        denoise_strength=12,
        clahe_clip=4.0,
        adaptive_block_size=15,
        adaptive_c=6,
        psm_mode=6,
    ),
}


# ─── Agent Orchestrator ──────────────────────────────────────────────────────

class OCRAgent:
    """
    The autonomous OCR Agent.
    Implements the full agentic cycle: Perceive → Decide → Act → Learn.
    """

    def __init__(
        self,
        memory: AgentMemory,
        logger: SafetyLogger,
        confidence_threshold: float = 55.0,
        max_retries: int = 3,
        autonomy_level: str = 'semi',  # 'semi' or 'full'
    ):
        self.memory = memory
        self.logger = logger
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.autonomy_level = autonomy_level
        self.current_analysis: Optional[ImageAnalysis] = None
        self.current_strategy: Optional[PreprocessingStrategy] = None
        self.retry_count = 0
        self.tried_strategies: List[str] = []

    # ─── PERCEIVE ─────────────────────────────────────────────────────────

    def perceive(self, image: Image.Image) -> ImageAnalysis:
        """
        Analyze the input image to understand its characteristics.
        This is the perception phase of the agent.
        """
        self.logger.log_perception("Starting image analysis...")

        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        analysis = ImageAnalysis()
        analysis.resolution = image.size

        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        analysis.blur_score = float(laplacian_var)

        # Noise estimation (standard deviation of Laplacian)
        noise = np.std(cv2.Laplacian(gray, cv2.CV_64F))
        analysis.noise_level = float(noise)

        # Contrast ratio
        analysis.contrast_ratio = float(gray.max() - gray.min()) / 255.0

        # Brightness
        analysis.brightness = float(np.mean(gray)) / 255.0

        # Color detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        analysis.has_color = float(np.mean(saturation)) > 30

        # Text density estimation (ratio of dark pixels)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        analysis.estimated_text_density = float(np.count_nonzero(binary)) / binary.size

        # Classify quality profile
        analysis.quality_profile = self._classify_quality(analysis)

        # Calculate overall quality score (0-100)
        analysis.quality_score = self._calculate_quality_score(analysis)

        self.current_analysis = analysis

        self.logger.log_perception(
            f"Image analysis complete: {analysis.quality_profile} quality",
            details=analysis.to_dict()
        )

        return analysis

    def _classify_quality(self, analysis: ImageAnalysis) -> str:
        """Classify image quality into a profile category."""
        if analysis.blur_score < 50:
            return 'blurry'
        elif analysis.noise_level > 40:
            return 'noisy'
        elif analysis.contrast_ratio < 0.4:
            return 'low_contrast'
        elif analysis.blur_score > 500 and analysis.contrast_ratio > 0.7:
            return 'clean'
        else:
            return 'mixed'

    def _calculate_quality_score(self, analysis: ImageAnalysis) -> float:
        """Calculate an overall quality score from 0-100."""
        score = 50.0  # Start at middle

        # Blur: higher is better (sharper)
        if analysis.blur_score > 500:
            score += 20
        elif analysis.blur_score > 200:
            score += 10
        elif analysis.blur_score < 50:
            score -= 20

        # Contrast: higher is better
        if analysis.contrast_ratio > 0.7:
            score += 15
        elif analysis.contrast_ratio > 0.5:
            score += 5
        elif analysis.contrast_ratio < 0.3:
            score -= 15

        # Resolution
        w, h = analysis.resolution
        pixels = w * h
        if pixels > 2_000_000:
            score += 10
        elif pixels < 500_000:
            score -= 10

        # Noise: lower is better
        if analysis.noise_level < 15:
            score += 5
        elif analysis.noise_level > 40:
            score -= 10

        return max(0, min(100, score))

    # ─── DECIDE ───────────────────────────────────────────────────────────

    def decide(self, analysis: Optional[ImageAnalysis] = None) -> PreprocessingStrategy:
        """
        Decide on the best preprocessing strategy based on image analysis.
        Checks memory for learned patterns first.
        """
        if analysis is None:
            analysis = self.current_analysis

        if analysis is None:
            # Fallback
            strategy = STRATEGIES['balanced']
            self.logger.log_decision(
                "Using default balanced strategy",
                "No image analysis available, falling back to balanced.",
            )
            self.current_strategy = strategy
            return strategy

        # Check memory for learned patterns
        memory_recommendation = self.memory.get_recommended_strategy(analysis.quality_profile)
        if memory_recommendation and memory_recommendation in STRATEGIES:
            if memory_recommendation not in self.tried_strategies:
                strategy = STRATEGIES[memory_recommendation]
                self.logger.log_decision(
                    f"Using memory-recommended strategy: {strategy.name}",
                    f"Memory indicates '{strategy.name}' works best for "
                    f"'{analysis.quality_profile}' quality images based on past conversions.",
                    details={'source': 'memory', 'profile': analysis.quality_profile},
                )
                self.current_strategy = strategy
                return strategy

        # Rule-based decision
        strategy = self._rule_based_decision(analysis)

        self.current_strategy = strategy
        return strategy

    def _rule_based_decision(self, analysis: ImageAnalysis) -> PreprocessingStrategy:
        """Make a rule-based strategy decision."""
        profile = analysis.quality_profile

        # Find an untried strategy
        strategy_map = {
            'clean': ['light', 'balanced', 'aggressive'],
            'noisy': ['aggressive', 'balanced', 'handwritten'],
            'blurry': ['aggressive', 'high_contrast', 'balanced'],
            'low_contrast': ['high_contrast', 'aggressive', 'balanced'],
            'mixed': ['balanced', 'handwritten', 'aggressive'],
        }

        candidates = strategy_map.get(profile, ['balanced', 'aggressive', 'light'])

        for candidate_name in candidates:
            if candidate_name not in self.tried_strategies:
                strategy = STRATEGIES[candidate_name]
                reasoning = (
                    f"Image quality is '{profile}'. "
                    f"Selected '{strategy.name}' strategy because: {strategy.description}. "
                    f"Blur={analysis.blur_score:.0f}, Noise={analysis.noise_level:.1f}, "
                    f"Contrast={analysis.contrast_ratio:.2f}."
                )
                self.logger.log_decision(
                    f"Selected strategy: {strategy.name}",
                    reasoning,
                    details={
                        'profile': profile,
                        'strategy': strategy.name,
                        'tried': self.tried_strategies.copy(),
                    },
                )
                return strategy

        # All strategies tried, use balanced as final fallback
        strategy = STRATEGIES['balanced']
        self.logger.log_decision(
            "All strategies exhausted, using balanced as fallback",
            "All available strategies have been tried. Using balanced as final attempt.",
        )
        return strategy

    # ─── ACT ──────────────────────────────────────────────────────────────

    def act(
        self,
        image: Image.Image,
        strategy: Optional[PreprocessingStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Execute the OCR pipeline with the selected strategy.
        Returns the full result including text blocks, confidence, etc.
        """
        from image_preprocessor import preprocess_adaptive, resize_for_ocr
        from ocr_engine import (
            extract_text_with_data,
            extract_text_simple,
            group_words_into_lines,
            lines_to_text_blocks,
        )
        from formatting_detector import detect_formatting

        if strategy is None:
            strategy = self.current_strategy or STRATEGIES['balanced']

        self.tried_strategies.append(strategy.name)

        self.logger.log_action(
            f"Executing OCR pipeline with '{strategy.name}' strategy",
            details=strategy.to_dict(),
        )

        # Step 1: Resize
        self.logger.log_action("Resizing image for optimal OCR performance")
        resized = resize_for_ocr(image)

        # Step 2: Adaptive Preprocessing
        self.logger.log_action(
            f"Applying '{strategy.name}' preprocessing",
            details={
                'denoise': strategy.denoise_strength,
                'clahe': strategy.clahe_clip,
                'threshold': strategy.do_threshold,
            },
        )
        processed = preprocess_adaptive(resized, strategy)

        # Step 3: OCR Extraction
        self.logger.log_action("Running Tesseract OCR extraction")
        words = extract_text_with_data(processed, psm_mode=strategy.psm_mode)

        # Step 4: Group and format
        self.logger.log_action("Grouping words into lines and blocks")
        lines = group_words_into_lines(words)
        blocks = lines_to_text_blocks(lines)

        # Step 5: Formatting detection
        self.logger.log_action("Detecting text formatting (headings, bold, alignment)")
        blocks = detect_formatting(image, blocks, image.size[0])

        # Calculate statistics
        total_words = sum(b['word_count'] for b in blocks)
        avg_confidence = (
            sum(b['confidence'] for b in blocks) / len(blocks)
            if blocks else 0
        )
        extracted_text = '\n'.join(b['text'] for b in blocks)
        headings_count = sum(1 for b in blocks if b.get('is_heading'))
        paragraphs_count = sum(1 for b in blocks if b.get('new_paragraph'))

        result = {
            'blocks': blocks,
            'text': extracted_text,
            'total_words': total_words,
            'avg_confidence': avg_confidence,
            'headings_count': headings_count,
            'paragraphs_count': paragraphs_count,
            'strategy_used': strategy.name,
            'processed_image': processed,
        }

        self.logger.log_action(
            f"OCR complete: {total_words} words, {avg_confidence:.1f}% confidence",
            details={
                'words': total_words,
                'confidence': round(avg_confidence, 1),
                'headings': headings_count,
                'paragraphs': paragraphs_count,
            },
        )

        return result

    # ─── QUALITY GATE ─────────────────────────────────────────────────────

    def check_quality(self, result: Dict[str, Any]) -> bool:
        """
        Quality gate: determine if the OCR result meets the threshold.
        Returns True if quality is acceptable.
        """
        confidence = result.get('avg_confidence', 0)
        word_count = result.get('total_words', 0)

        passed = confidence >= self.confidence_threshold and word_count > 0

        if passed:
            self.logger.log_action(
                f"Quality gate PASSED: {confidence:.1f}% ≥ {self.confidence_threshold}%",
                details={'confidence': confidence, 'threshold': self.confidence_threshold},
            )
        else:
            reasons = []
            if confidence < self.confidence_threshold:
                reasons.append(
                    f"confidence {confidence:.1f}% < threshold {self.confidence_threshold}%"
                )
            if word_count == 0:
                reasons.append("no words extracted")

            self.logger.log_warning(
                f"Quality gate FAILED: {', '.join(reasons)}",
                details={
                    'confidence': confidence,
                    'threshold': self.confidence_threshold,
                    'word_count': word_count,
                },
            )

        return passed

    def should_retry(self) -> bool:
        """Determine if the agent should retry with a different strategy."""
        can_retry = self.retry_count < self.max_retries
        has_untried = any(
            name not in self.tried_strategies for name in STRATEGIES
        )

        should = can_retry and has_untried

        if should:
            self.retry_count += 1
            self.logger.log_decision(
                f"Retrying with different strategy (attempt {self.retry_count}/{self.max_retries})",
                f"Quality gate failed. {len(STRATEGIES) - len(self.tried_strategies)} "
                f"untried strategies remain.",
            )
        elif not can_retry:
            self.logger.log_warning(
                f"Max retries ({self.max_retries}) reached. Using best result so far."
            )
        elif not has_untried:
            self.logger.log_warning(
                "All strategies have been tried. Using best result so far."
            )

        return should

    # ─── LEARN ────────────────────────────────────────────────────────────

    def learn(
        self,
        filename: str,
        result: Dict[str, Any],
        success: bool,
        human_override: bool = False,
    ):
        """
        Learn from the conversion outcome.
        Updates both short-term and long-term memory.
        """
        analysis = self.current_analysis or ImageAnalysis()

        record = ConversionRecord(
            timestamp=datetime.now().isoformat(),
            filename=filename,
            image_quality=analysis.to_dict(),
            strategy_used=result.get('strategy_used', 'unknown'),
            preprocessing_params=(
                self.current_strategy.to_dict() if self.current_strategy else {}
            ),
            ocr_confidence=result.get('avg_confidence', 0),
            word_count=result.get('total_words', 0),
            retry_count=self.retry_count,
            success=success,
            human_override=human_override,
        )

        self.memory.record_conversion(record)

        self.logger.log_learning(
            f"Recorded conversion outcome: {'success' if success else 'partial'}",
            details={
                'confidence': record.ocr_confidence,
                'strategy': record.strategy_used,
                'retries': record.retry_count,
                'human_override': human_override,
            },
        )

    # ─── FULL PIPELINE ────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        image: Image.Image,
        filename: str,
        use_formatted: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete agentic pipeline:
        Perceive → Decide → Act → (Quality Gate → Retry?) → Learn

        Returns the best result achieved.
        """
        # Reset state for new image
        self.retry_count = 0
        self.tried_strategies = []
        self.logger.clear()

        # PERCEIVE
        analysis = self.perceive(image)

        # Safety check for very large images
        w, h = analysis.resolution
        if w * h > 20_000_000:
            self.logger.log_safety(
                "Very large image detected. May cause memory issues.",
                details={'resolution': f"{w}×{h}", 'pixels': w * h},
            )

        best_result = None
        best_confidence = 0

        while True:
            # DECIDE
            strategy = self.decide(analysis)

            # ACT
            if use_formatted:
                result = self.act(image, strategy)
            else:
                # Plain text mode
                from image_preprocessor import preprocess_adaptive, resize_for_ocr
                from ocr_engine import extract_text_simple

                resized = resize_for_ocr(image)
                processed = preprocess_adaptive(resized, strategy)
                text = extract_text_simple(processed)

                result = {
                    'blocks': [],
                    'text': text,
                    'total_words': len(text.split()) if text else 0,
                    'avg_confidence': 70 if text else 0,  # Estimate
                    'headings_count': 0,
                    'paragraphs_count': text.count('\n\n') + 1 if text else 0,
                    'strategy_used': strategy.name,
                    'processed_image': processed,
                }

            # Track best result
            current_conf = result.get('avg_confidence', 0)
            if current_conf > best_confidence:
                best_confidence = current_conf
                best_result = result

            # QUALITY GATE
            if self.check_quality(result):
                break  # Quality is good enough

            # Check if we should retry
            if not self.should_retry():
                break  # Max retries or all strategies tried

        # Use best result
        final_result = best_result or result

        # LEARN
        success = final_result.get('avg_confidence', 0) >= self.confidence_threshold
        self.learn(filename, final_result, success)

        # Add agent metadata to result
        final_result['agent_metadata'] = {
            'analysis': analysis.to_dict(),
            'strategy': (
                self.current_strategy.to_dict() if self.current_strategy else {}
            ),
            'retries': self.retry_count,
            'quality_passed': success,
            'decisions': self.logger.get_decision_trail(),
            'explainability': self.logger.get_explainability_report(),
        }

        return final_result

    def apply_human_override(self, strategy_name: str):
        """Apply a human override to change the strategy."""
        if strategy_name in STRATEGIES:
            old_name = self.current_strategy.name if self.current_strategy else 'none'
            self.current_strategy = STRATEGIES[strategy_name]
            self.logger.log_override(
                f"Human override: changed strategy from '{old_name}' to '{strategy_name}'",
                details={'old': old_name, 'new': strategy_name},
            )
        else:
            self.logger.log_warning(
                f"Invalid strategy name for override: {strategy_name}",
                details={'available': list(STRATEGIES.keys())},
            )

    def reset(self):
        """Reset agent state for a new conversion."""
        self.current_analysis = None
        self.current_strategy = None
        self.retry_count = 0
        self.tried_strategies = []
