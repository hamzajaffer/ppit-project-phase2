"""
Memory System Module
Provides short-term and long-term memory for the OCR Agent.
Tracks conversion history, image characteristics, and learned strategies.
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ConversionRecord:
    """A single conversion record stored in memory."""
    timestamp: str
    filename: str
    image_quality: Dict[str, Any]
    strategy_used: str
    preprocessing_params: Dict[str, Any]
    ocr_confidence: float
    word_count: int
    retry_count: int
    success: bool
    human_override: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> 'ConversionRecord':
        return ConversionRecord(**data)


class ShortTermMemory:
    """
    Short-term memory for the current session.
    Stores recent conversion records and active context.
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.records: List[ConversionRecord] = []
        self.active_context: Dict[str, Any] = {}
        self.session_start = datetime.now().isoformat()

    def add_record(self, record: ConversionRecord):
        """Add a conversion record, evicting oldest if at capacity."""
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records.pop(0)

    def set_context(self, key: str, value: Any):
        """Set a context variable for the current operation."""
        self.active_context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.active_context.get(key, default)

    def clear_context(self):
        """Clear the active context."""
        self.active_context = {}

    def get_recent_records(self, n: int = 5) -> List[ConversionRecord]:
        """Get the N most recent conversion records."""
        return self.records[-n:]

    def get_average_confidence(self) -> float:
        """Get average OCR confidence across all records."""
        if not self.records:
            return 0.0
        return sum(r.ocr_confidence for r in self.records) / len(self.records)

    def get_best_strategy_for_quality(self, quality_profile: str) -> Optional[str]:
        """
        Look up what strategy worked best for similar image quality.
        Returns the strategy name or None if no data available.
        """
        matching = [
            r for r in self.records
            if r.success and r.image_quality.get('quality_profile') == quality_profile
        ]
        if not matching:
            return None

        # Return the strategy with highest confidence
        best = max(matching, key=lambda r: r.ocr_confidence)
        return best.strategy_used

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        if not self.records:
            return {
                'session_start': self.session_start,
                'total_conversions': 0,
                'avg_confidence': 0,
                'success_rate': 0,
                'strategies_used': [],
                'total_retries': 0,
                'human_overrides': 0,
            }

        successful = [r for r in self.records if r.success]
        strategies = list(set(r.strategy_used for r in self.records))

        return {
            'session_start': self.session_start,
            'total_conversions': len(self.records),
            'avg_confidence': self.get_average_confidence(),
            'success_rate': len(successful) / len(self.records) * 100,
            'strategies_used': strategies,
            'total_retries': sum(r.retry_count for r in self.records),
            'human_overrides': sum(1 for r in self.records if r.human_override),
        }


class LongTermMemory:
    """
    Long-term memory that persists across sessions via Streamlit session state.
    Stores strategy effectiveness data and learned patterns.
    """

    def __init__(self):
        self.strategy_scores: Dict[str, List[float]] = {}
        self.quality_strategy_map: Dict[str, str] = {}
        self.total_conversions: int = 0
        self.total_successful: int = 0

    def learn_from_record(self, record: ConversionRecord):
        """Update long-term knowledge from a conversion record."""
        self.total_conversions += 1
        if record.success:
            self.total_successful += 1

        # Track strategy effectiveness
        strategy = record.strategy_used
        if strategy not in self.strategy_scores:
            self.strategy_scores[strategy] = []
        self.strategy_scores[strategy].append(record.ocr_confidence)

        # Update quality → strategy mapping
        quality_profile = record.image_quality.get('quality_profile', 'unknown')
        if record.success and record.ocr_confidence > 60:
            current_best = self.quality_strategy_map.get(quality_profile)
            if current_best is None:
                self.quality_strategy_map[quality_profile] = strategy
            else:
                # Update if this strategy performed better
                current_avg = self._get_strategy_avg(current_best)
                new_avg = self._get_strategy_avg(strategy)
                if new_avg > current_avg:
                    self.quality_strategy_map[quality_profile] = strategy

    def _get_strategy_avg(self, strategy: str) -> float:
        """Get average confidence for a strategy."""
        scores = self.strategy_scores.get(strategy, [])
        return sum(scores) / len(scores) if scores else 0.0

    def get_recommended_strategy(self, quality_profile: str) -> Optional[str]:
        """Get the recommended strategy for a quality profile."""
        return self.quality_strategy_map.get(quality_profile)

    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all strategies."""
        stats = {}
        for strategy, scores in self.strategy_scores.items():
            stats[strategy] = {
                'times_used': len(scores),
                'avg_confidence': sum(scores) / len(scores) if scores else 0,
                'min_confidence': min(scores) if scores else 0,
                'max_confidence': max(scores) if scores else 0,
            }
        return stats

    def to_dict(self) -> dict:
        """Serialize for storage in session state."""
        return {
            'strategy_scores': self.strategy_scores,
            'quality_strategy_map': self.quality_strategy_map,
            'total_conversions': self.total_conversions,
            'total_successful': self.total_successful,
        }

    @staticmethod
    def from_dict(data: dict) -> 'LongTermMemory':
        """Deserialize from session state."""
        mem = LongTermMemory()
        mem.strategy_scores = data.get('strategy_scores', {})
        mem.quality_strategy_map = data.get('quality_strategy_map', {})
        mem.total_conversions = data.get('total_conversions', 0)
        mem.total_successful = data.get('total_successful', 0)
        return mem


class AgentMemory:
    """
    Combined memory system for the OCR Agent.
    Manages both short-term (session) and long-term (persistent) memory.
    """

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def record_conversion(self, record: ConversionRecord):
        """Record a conversion in both short-term and long-term memory."""
        self.short_term.add_record(record)
        self.long_term.learn_from_record(record)

    def get_recommended_strategy(self, quality_profile: str) -> Optional[str]:
        """
        Get the best strategy recommendation.
        Checks short-term memory first (recent context), then long-term.
        """
        # Check short-term memory first (recent session context)
        stm_recommendation = self.short_term.get_best_strategy_for_quality(quality_profile)
        if stm_recommendation:
            return stm_recommendation

        # Fall back to long-term learned patterns
        return self.long_term.get_recommended_strategy(quality_profile)

    def get_full_summary(self) -> Dict[str, Any]:
        """Get a comprehensive memory summary."""
        return {
            'session': self.short_term.get_session_summary(),
            'long_term': {
                'total_conversions': self.long_term.total_conversions,
                'total_successful': self.long_term.total_successful,
                'success_rate': (
                    self.long_term.total_successful / self.long_term.total_conversions * 100
                    if self.long_term.total_conversions > 0 else 0
                ),
                'strategy_stats': self.long_term.get_strategy_stats(),
                'learned_mappings': self.long_term.quality_strategy_map,
            }
        }
