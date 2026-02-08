"""State 스키마 테스트"""

import pytest
from addie_agent.state import (
    ADDIEState,
    create_initial_state,
    ScenarioInput,
    ToolCall,
)


def test_create_initial_state():
    """초기 상태 생성 테스트"""
    scenario = {
        "scenario_id": "TEST-001",
        "title": "테스트 시나리오",
        "context": {
            "target_audience": "신입사원",
            "duration": "2시간",
            "learning_environment": "대면 교육",
        },
        "learning_goals": ["목표1", "목표2"],
    }

    state = create_initial_state(scenario)

    assert state["scenario"] == scenario
    assert state["current_phase"] == "analysis"
    assert state["errors"] == []
    assert state["tool_calls"] == []
    assert state["reasoning_steps"] == []


def test_tool_call_structure():
    """ToolCall 구조 테스트"""
    tool_call: ToolCall = {
        "step": 1,
        "tool": "analyze_learner",
        "args": {"target_audience": "신입사원"},
        "result": "학습자 분석 완료",
        "timestamp": "2025-01-13T00:00:00",
        "duration_ms": 1000,
        "success": True,
    }

    assert tool_call["step"] == 1
    assert tool_call["tool"] == "analyze_learner"
    assert tool_call["success"] is True


def test_addie_state_phases():
    """ADDIE 단계 타입 테스트"""
    valid_phases = ["analysis", "design", "development", "implementation", "evaluation", "complete"]

    for phase in valid_phases:
        state: ADDIEState = {
            "scenario": {},
            "current_phase": phase,
            "errors": [],
            "tool_calls": [],
            "reasoning_steps": [],
        }
        assert state["current_phase"] == phase
