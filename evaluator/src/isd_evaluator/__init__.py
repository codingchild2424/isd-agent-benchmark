"""
ISD Evaluator: 교수설계 Agent 평가 시스템

ADDIE 루브릭 기반 평가와 궤적 평가를 통해
교수설계 Agent의 산출물과 생성 과정을 평가합니다.
"""

__version__ = "0.2.0"

# Lazy imports: openai 의존 모듈은 실제 사용 시점에 import
# ContextWeightAdjuster는 openai 미사용이므로 즉시 import
from isd_evaluator.metrics.context_weights import ContextWeightAdjuster

__all__ = [
    "ADDIERubricEvaluator",
    "ADDIEScore",
    "TrajectoryEvaluator",
    "TrajectoryScore",
    "CompositeEvaluator",
    "CompositeScore",
    "ContextWeightAdjuster",
    "AgentRunner",
    "ComparisonReporter",
    "__version__",
]


def __getattr__(name: str):
    """Lazy import for openai-dependent modules."""
    if name == "ADDIERubricEvaluator":
        from isd_evaluator.metrics.addie_rubric import ADDIERubricEvaluator
        return ADDIERubricEvaluator
    elif name == "ADDIEScore":
        from isd_evaluator.metrics.addie_rubric import ADDIEScore
        return ADDIEScore
    elif name == "TrajectoryEvaluator":
        from isd_evaluator.metrics.trajectory import TrajectoryEvaluator
        return TrajectoryEvaluator
    elif name == "TrajectoryScore":
        from isd_evaluator.metrics.trajectory import TrajectoryScore
        return TrajectoryScore
    elif name == "CompositeEvaluator":
        from isd_evaluator.metrics.composite import CompositeEvaluator
        return CompositeEvaluator
    elif name == "CompositeScore":
        from isd_evaluator.metrics.composite import CompositeScore
        return CompositeScore
    elif name == "AgentRunner":
        from isd_evaluator.runners import AgentRunner
        return AgentRunner
    elif name == "ComparisonReporter":
        from isd_evaluator.reporters import ComparisonReporter
        return ComparisonReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
