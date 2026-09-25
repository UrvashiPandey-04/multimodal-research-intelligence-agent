"""
Lightweight structured logging module for the Multimodal Research Intelligence Agent.
Ensures clean event logging without leaking API keys, secrets, or full raw document contents.
"""

import logging
import sys
from typing import Dict, Any, Optional

# Global logger setup
_LOGGER = logging.getLogger("research_agent")
_LOGGER.setLevel(logging.INFO)

if not _LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)


def get_logger() -> logging.Logger:
    """Return the global research_agent logger instance."""
    return _LOGGER


def log_event(event_name: str, details: Optional[Dict[str, Any]] = None, level: int = logging.INFO) -> None:
    """
    Log a structured event safely without exposing API keys or secret environment variables.

    Args:
        event_name: Short name of the event (e.g., 'query_received', 'tools_selected').
        details: Optional dictionary of metadata associated with the event.
        level: Logging level (default: logging.INFO).
    """
    safe_details = {}
    if details:
        for k, v in details.items():
            # Filter sensitive keywords
            if any(secret in k.lower() for secret in ["key", "secret", "password", "token", "auth"]):
                safe_details[k] = "[REDACTED]"
            elif isinstance(v, str) and len(v) > 300:
                safe_details[k] = v[:300] + "... [TRUNCATED]"
            else:
                safe_details[k] = v

    msg = f"EVENT: {event_name}"
    if safe_details:
        msg += f" | Details: {safe_details}"

    _LOGGER.log(level, msg)
