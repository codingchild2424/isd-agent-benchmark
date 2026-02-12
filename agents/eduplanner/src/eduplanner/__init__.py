"""
EduPlanner: 3-Agent 기반 교수설계 시스템

3개의 에이전트(Evaluator, Optimizer, Analyst)가 협업하여
맞춤형 교수설계를 생성하는 시스템입니다.

References:
    - EduPlanner: LLM-Based Multi-Agent Systems for Customized and
      Intelligent Instructional Design (Zhang et al., 2025)
"""

# Python 3.14 + LangChain Pydantic V1 호환성 경고 필터링
# LangChain Core가 내부적으로 pydantic.v1을 사용하여 Python 3.14에서 경고 발생
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

__version__ = "0.1.0"

from eduplanner.agents.main import EduPlannerAgent
from eduplanner.models.schemas import ScenarioInput, ADDIEOutput, AgentResult

__all__ = [
    "EduPlannerAgent",
    "ScenarioInput",
    "ADDIEOutput",
    "AgentResult",
    "__version__",
]
