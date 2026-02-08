"""
RPISD Agent: Rapid Prototyping Instructional System Design

래피드 프로토타이핑 기반 교수설계 에이전트
- 이중 루프: 설계↔사용성평가 (프로토타입), 개발↔사용성평가 (최종)
- 프로토타입 버전 관리
- 다중 피드백 통합 (의뢰인/전문가/학습자)
"""

from rpisd_agent.agent import RPISDAgent
from rpisd_agent.state import RPISDState, create_initial_state

__version__ = "0.1.0"
__all__ = ["RPISDAgent", "RPISDState", "create_initial_state"]
