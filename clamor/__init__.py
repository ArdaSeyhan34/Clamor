"""Clamor: turn the noise of customer feedback into a prioritized, evidence-backed roadmap.

Quick start::

    from clamor import analyze
    result = analyze("feedback.csv", accounts="accounts.csv", releases="releases.csv")
    result.roadmap.head()
"""

from .config import PRESETS, Config, Weights
from .pipeline import Analysis, analyze

__all__ = ["PRESETS", "Analysis", "Config", "Weights", "analyze"]
__version__ = "0.1.0"
