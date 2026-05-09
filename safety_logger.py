"""
Safety & Logging Module
Provides structured logging, audit trails, explainability reports,
and safety mechanisms for the OCR Agent.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    INFO = "INFO"
    DECISION = "DECISION"
    ACTION = "ACTION"
    WARNING = "WARNING"
    ERROR = "ERROR"
    OVERRIDE = "OVERRIDE"
    SAFETY = "SAFETY"


class AgentPhase(Enum):
    PERCEIVE = "👁️ Perceive"
    DECIDE = "🧠 Decide"
    ACT = "⚡ Act"
    LEARN = "📝 Learn"
    SAFETY = "🛡️ Safety"
    HUMAN = "👤 Human"


@dataclass
class LogEntry:
    """A single log entry in the agent's audit trail."""
    timestamp: str
    level: str
    phase: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SafetyLogger:
    """
    Comprehensive logging and safety system for the OCR Agent.
    Provides audit trail, explainability, and safety mechanisms.
    """

    def __init__(self):
        self.logs: List[LogEntry] = []
        self.decisions: List[Dict[str, Any]] = []
        self.overrides: List[Dict[str, Any]] = []
        self.safety_flags: List[Dict[str, Any]] = []
        self._start_time = time.time()

    def log(
        self,
        level: LogLevel,
        phase: AgentPhase,
        message: str,
        details: Dict[str, Any] = None,
        reasoning: str = ""
    ):
        """Add a log entry."""
        entry = LogEntry(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            level=level.value,
            phase=phase.value,
            message=message,
            details=details or {},
            reasoning=reasoning,
        )
        self.logs.append(entry)

        # Track decisions separately
        if level == LogLevel.DECISION:
            self.decisions.append(entry.to_dict())

        # Track overrides
        if level == LogLevel.OVERRIDE:
            self.overrides.append(entry.to_dict())

    def log_perception(self, message: str, details: Dict[str, Any] = None):
        """Log a perception event."""
        self.log(LogLevel.INFO, AgentPhase.PERCEIVE, message, details)

    def log_decision(self, message: str, reasoning: str, details: Dict[str, Any] = None):
        """Log a decision with reasoning."""
        self.log(LogLevel.DECISION, AgentPhase.DECIDE, message, details, reasoning)

    def log_action(self, message: str, details: Dict[str, Any] = None):
        """Log an action taken."""
        self.log(LogLevel.ACTION, AgentPhase.ACT, message, details)

    def log_learning(self, message: str, details: Dict[str, Any] = None):
        """Log a learning event."""
        self.log(LogLevel.INFO, AgentPhase.LEARN, message, details)

    def log_override(self, message: str, details: Dict[str, Any] = None):
        """Log a human override."""
        self.log(LogLevel.OVERRIDE, AgentPhase.HUMAN, message, details)

    def log_warning(self, message: str, details: Dict[str, Any] = None):
        """Log a warning."""
        self.log(LogLevel.WARNING, AgentPhase.SAFETY, message, details)

    def log_safety(self, message: str, details: Dict[str, Any] = None):
        """Log a safety flag."""
        self.log(LogLevel.SAFETY, AgentPhase.SAFETY, message, details)
        self.safety_flags.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'message': message,
            'details': details or {},
        })

    def get_decision_trail(self) -> List[Dict[str, Any]]:
        """Get all decisions made by the agent."""
        return self.decisions

    def get_overrides(self) -> List[Dict[str, Any]]:
        """Get all human overrides."""
        return self.overrides

    def get_safety_flags(self) -> List[Dict[str, Any]]:
        """Get all safety flags raised."""
        return self.safety_flags

    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Get all log entries as dicts."""
        return [entry.to_dict() for entry in self.logs]

    def get_logs_by_phase(self, phase: AgentPhase) -> List[Dict[str, Any]]:
        """Get logs filtered by phase."""
        return [
            entry.to_dict() for entry in self.logs
            if entry.phase == phase.value
        ]

    def get_explainability_report(self) -> Dict[str, Any]:
        """
        Generate a human-readable explainability report.
        Summarizes all decisions and their reasoning.
        """
        elapsed = time.time() - self._start_time

        report = {
            'total_time_seconds': round(elapsed, 2),
            'total_log_entries': len(self.logs),
            'total_decisions': len(self.decisions),
            'total_overrides': len(self.overrides),
            'total_safety_flags': len(self.safety_flags),
            'decisions_summary': [],
            'safety_summary': [],
        }

        for decision in self.decisions:
            report['decisions_summary'].append({
                'time': decision['timestamp'],
                'what': decision['message'],
                'why': decision['reasoning'],
            })

        for flag in self.safety_flags:
            report['safety_summary'].append({
                'time': flag['timestamp'],
                'issue': flag['message'],
            })

        return report

    def get_formatted_log(self) -> str:
        """Get a formatted string of all logs for display."""
        lines = []
        for entry in self.logs:
            icon = {
                'INFO': 'ℹ️',
                'DECISION': '🧠',
                'ACTION': '⚡',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'OVERRIDE': '👤',
                'SAFETY': '🛡️',
            }.get(entry.level, '📋')

            line = f"`{entry.timestamp}` {icon} **{entry.phase}** — {entry.message}"
            if entry.reasoning:
                line += f"\n   > *Reasoning: {entry.reasoning}*"
            lines.append(line)

        return "\n\n".join(lines)

    def get_data_handling_info(self) -> Dict[str, str]:
        """
        Return transparency information about data handling.
        This supports ethical/legal requirements of Phase 2.
        """
        return {
            'data_storage': 'Images are processed in-memory only. No images are stored on any server.',
            'data_retention': 'Session data is cleared when you close the browser tab.',
            'data_sharing': 'No data is shared with third parties. All processing is local.',
            'ocr_engine': 'Text extraction uses Tesseract OCR (open-source, offline).',
            'privacy': 'No personal data is collected. No cookies or tracking.',
            'user_rights': 'You retain full ownership of all uploaded images and generated documents.',
            'security': 'All processing happens in an isolated session. No cross-user data access.',
        }

    def clear(self):
        """Clear all logs."""
        self.logs = []
        self.decisions = []
        self.overrides = []
        self.safety_flags = []
        self._start_time = time.time()
