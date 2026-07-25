"""Base agent definition and common payload data structures."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Standardized output structure produced by any AI Agent execution."""

    agent_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_time_ms: float = 0.0


class BaseAgent(ABC):
    """Abstract base class for all AI Agents in the Fatigue Fingerprinting framework."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agents.{agent_name}")

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> AgentResult:
        """Process the given context and return an AgentResult.

        Args:
            context: Dictionary containing inputs such as raw signals,
                quality reports, model predictions, SHAP vectors, or subject profiles.

        Returns:
            AgentResult: Structured output from the agent's execution.
        """
        pass

    def _create_result(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        narrative: str = "",
        warnings: Optional[List[str]] = None,
        execution_time_ms: float = 0.0,
    ) -> AgentResult:
        """Utility method to construct a standard AgentResult instance."""
        return AgentResult(
            agent_name=self.agent_name,
            success=success,
            data=data or {},
            narrative=narrative,
            warnings=warnings or [],
            execution_time_ms=execution_time_ms,
        )
