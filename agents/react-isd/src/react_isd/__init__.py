"""
ReAct-ISD: LangGraph ReAct 패턴 기반 교수설계 에이전트

ADDIE 각 단계를 도구로 분리하여 에이전트가
자율적으로 호출하는 ReAct 패턴을 구현합니다.
"""

__version__ = "0.1.0"

from react_isd.agent import ReActISDAgent

__all__ = ["ReActISDAgent", "__version__"]
