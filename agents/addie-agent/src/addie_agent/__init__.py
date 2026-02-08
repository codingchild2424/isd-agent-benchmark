"""
ADDIE Agent: 순차적 교수설계 에이전트

ADDIE 모형(Analysis-Design-Development-Implementation-Evaluation)의
선형적/순차적 프로세스를 LangGraph StateGraph로 구현한 교수설계 에이전트입니다.
"""

from addie_agent.agent import ADDIEAgent
from addie_agent.state import ADDIEState

__version__ = "0.1.0"
__all__ = ["ADDIEAgent", "ADDIEState"]
