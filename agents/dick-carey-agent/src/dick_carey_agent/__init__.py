"""
Dick & Carey Agent: 체제적 교수설계(Systems Approach) 10단계 에이전트

Dick & Carey 모형의 10단계 프로세스를 LangGraph StateGraph로 구현합니다.
형성평가-수정 피드백 루프를 통해 지속적인 개선을 수행합니다.
"""

__version__ = "0.1.0"

from dick_carey_agent.agent import DickCareyAgent
from dick_carey_agent.state import DickCareyState, create_initial_state

__all__ = ["DickCareyAgent", "DickCareyState", "create_initial_state", "__version__"]
