"""
State 스키마 테스트
"""

import pytest
from rpisd_agent.state import (
    RPISDState,
    ToolCall,
    PrototypeVersion,
    UsabilityFeedback,
    create_initial_state,
    record_prototype_version,
    map_to_addie_output,
)


class TestCreateInitialState:
    """create_initial_state 함수 테스트"""

    def test_basic_scenario(self):
        """기본 시나리오로 초기 상태 생성"""
        scenario = {
            "scenario_id": "TEST-001",
            "title": "테스트 교육",
            "learning_goals": ["목표 1", "목표 2"],
        }

        state = create_initial_state(scenario)

        assert state["scenario"] == scenario
        assert state["current_phase"] == "kickoff"
        assert state["prototype_iteration"] == 0
        assert state["development_iteration"] == 0
        assert state["max_iterations"] == 3
        assert state["quality_threshold"] == 0.8
        assert state["current_quality"] == 0.0
        assert state["errors"] == []
        assert state["tool_calls"] == []
        assert state["reasoning_steps"] == []

    def test_empty_results(self):
        """초기 상태의 결과들이 빈 dict인지 확인"""
        state = create_initial_state({"scenario_id": "TEST"})

        assert state["kickoff_result"] == {}
        assert state["analysis_result"] == {}
        assert state["design_result"] == {}
        assert state["development_result"] == {}
        assert state["implementation_result"] == {}
        assert state["prototype_versions"] == []


class TestRecordPrototypeVersion:
    """record_prototype_version 함수 테스트"""

    def test_first_version(self):
        """첫 번째 프로토타입 버전 기록"""
        state = create_initial_state({"scenario_id": "TEST"})
        content = {"modules": [{"id": "M1"}]}
        feedback = [{"score": 0.7}]

        version = record_prototype_version(state, content, feedback, 0.75)

        assert version["version"] == 1
        assert version["content"] == content
        assert version["feedback"] == feedback
        assert version["quality_score"] == 0.75
        assert "timestamp" in version

    def test_subsequent_versions(self):
        """연속 버전 기록"""
        state = create_initial_state({"scenario_id": "TEST"})
        state["prototype_versions"] = [{"version": 1}, {"version": 2}]

        version = record_prototype_version(state, {}, [], 0.8)

        assert version["version"] == 3


class TestMapToAddieOutput:
    """map_to_addie_output 함수 테스트"""

    def test_empty_state(self):
        """빈 상태에서 ADDIE 출력 생성"""
        state = create_initial_state({"scenario_id": "TEST"})

        output = map_to_addie_output(state)

        assert "analysis" in output
        assert "design" in output
        assert "development" in output
        assert "implementation" in output
        assert "evaluation" in output

    def test_full_state(self):
        """완전한 상태에서 ADDIE 출력 생성"""
        state = create_initial_state({"scenario_id": "TEST"})
        state["kickoff_result"] = {"scope": {"objective": "목표"}}
        state["analysis_result"] = {
            "learner_characteristics": {"target_audience": "신입사원"},
            "gap_analysis": {"gaps": ["gap1"]},
        }
        state["design_result"] = {
            "objectives": [{"id": "LO-001", "statement": "목표 1"}],
        }
        state["prototype_versions"] = [
            {"version": 1, "quality_score": 0.7},
            {"version": 2, "quality_score": 0.85},
        ]
        state["prototype_iteration"] = 2
        state["development_iteration"] = 1
        state["current_quality"] = 0.85

        output = map_to_addie_output(state)

        # Analysis
        assert output["analysis"]["learner_analysis"]["target_audience"] == "신입사원"
        assert output["analysis"]["task_analysis"]["gap_analysis"]["gaps"] == ["gap1"]

        # Design
        assert len(output["design"]["learning_objectives"]) == 1
        assert len(output["design"]["prototype_history"]) == 2

        # Evaluation
        assert output["evaluation"]["iteration_summary"]["prototype_iterations"] == 2
        assert output["evaluation"]["iteration_summary"]["final_quality_score"] == 0.85


class TestTypeDefinitions:
    """TypedDict 정의 테스트"""

    def test_tool_call(self):
        """ToolCall 구조"""
        tool_call: ToolCall = {
            "step": 1,
            "tool": "test_tool",
            "args": {"arg1": "value1"},
            "result": "성공",
            "timestamp": "2024-01-01T00:00:00",
            "duration_ms": 100,
            "success": True,
        }

        assert tool_call["step"] == 1
        assert tool_call["tool"] == "test_tool"

    def test_prototype_version(self):
        """PrototypeVersion 구조"""
        version: PrototypeVersion = {
            "version": 1,
            "content": {"modules": []},
            "feedback": [],
            "quality_score": 0.75,
            "timestamp": "2024-01-01T00:00:00",
        }

        assert version["version"] == 1
        assert version["quality_score"] == 0.75

    def test_usability_feedback(self):
        """UsabilityFeedback 구조"""
        feedback: UsabilityFeedback = {
            "client_feedback": {"score": 0.8},
            "expert_feedback": {"score": 0.85},
            "learner_feedback": {"score": 0.7},
            "aggregated_score": 0.78,
            "improvement_areas": ["영역1"],
            "recommendations": ["권고1"],
        }

        assert feedback["aggregated_score"] == 0.78
