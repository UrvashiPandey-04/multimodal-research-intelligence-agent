"""
Research agent evaluation framework package.
"""

from src.evaluation.schema import EvaluationItem, SingleEvaluationResult, AggregateMetrics
from src.evaluation.dataset import build_30_question_dataset, save_dataset_to_file, load_dataset_from_file
from src.evaluation.metrics import MetricsEngine
from src.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationItem",
    "SingleEvaluationResult",
    "AggregateMetrics",
    "build_30_question_dataset",
    "save_dataset_to_file",
    "load_dataset_from_file",
    "MetricsEngine",
    "EvaluationRunner",
]
