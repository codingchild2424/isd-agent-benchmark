"""
State 스키마 테스트

Dick & Carey State 초기화 및 유효성 검증 테스트
"""

import pytest
from dick_carey_agent.state import (
    DickCareyState,
    ScenarioInput,
    create_initial_state,
    map_to_addie_output,
)


class TestCreateInitialState:
    """create_initial_state 함수 테스트"""

    def test_basic_creation(self):
        """기본 초기 상태 생성 테스트"""
        scenario = ScenarioInput(
            scenario_id="TEST-001",
            title="테스트 시나리오",
            context={
                "target_audience": "신입사원",
                "duration": "2시간",
                "learning_environment": "온라인",
            },
            learning_goals=["목표 1", "목표 2"],
            domain="기업교육",
            difficulty="easy",
        )

        state = create_initial_state(scenario)

        assert state["scenario"]["scenario_id"] == "TEST-001"
        assert state["current_phase"] == "goal"
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 3
        assert state["quality_threshold"] == 7.0
        assert state["revision_triggered"] == False
        assert state["errors"] == []
        assert state["tool_calls"] == []
        assert state["reasoning_steps"] == []

    def test_empty_results(self):
        """초기 상태의 결과 필드가 비어있는지 테스트"""
        scenario = ScenarioInput(scenario_id="TEST-002")
        state = create_initial_state(scenario)

        assert state["goal"] == {}
        assert state["instructional_analysis"] == {}
        assert state["learner_context"] == {}
        assert state["performance_objectives"] == {}
        assert state["assessment_instruments"] == {}
        assert state["instructional_strategy"] == {}
        assert state["instructional_materials"] == {}
        assert state["formative_evaluation"] == {}
        assert state["revision_log"] == []
        assert state["summative_evaluation"] == {}

    def test_metadata_initialization(self):
        """메타데이터 초기화 테스트"""
        scenario = ScenarioInput(scenario_id="TEST-003")
        state = create_initial_state(scenario)

        assert "metadata" in state
        assert state["metadata"]["agent_version"] == "0.1.0"
        assert state["metadata"]["tool_calls_count"] == 0
        assert state["metadata"]["iteration_count"] == 0


class TestMapToAddieOutput:
    """map_to_addie_output 함수 테스트"""

    def test_empty_state_mapping(self):
        """빈 상태 매핑 테스트"""
        state = DickCareyState(
            scenario={},
            goal={},
            instructional_analysis={},
            learner_context={},
            performance_objectives={},
            assessment_instruments={},
            instructional_strategy={},
            instructional_materials={},
            formative_evaluation={},
            revision_log=[],
            summative_evaluation={},
        )

        addie = map_to_addie_output(state)

        assert "analysis" in addie
        assert "design" in addie
        assert "development" in addie
        assert "implementation" in addie
        assert "evaluation" in addie

    def test_mapping_structure(self):
        """매핑 구조 테스트"""
        state = DickCareyState(
            scenario={},
            goal={"goal_statement": "테스트 목표"},
            instructional_analysis={"task_type": "조합형"},
            learner_context={
                "learner": {"target_audience": "신입사원"},
                "context": {"learning_context": "온라인"},
            },
            performance_objectives={
                "terminal_objective": {"id": "PO-T01"},
                "enabling_objectives": [{"id": "PO-001"}],
            },
            assessment_instruments={
                "post_test": [{"id": "POST-001"}],
            },
            instructional_strategy={
                "delivery_method": "온라인",
            },
            instructional_materials={
                "instructor_guide": {"title": "가이드"},
                "learner_materials": [{"title": "워크북"}],
            },
            formative_evaluation={"quality_score": 7.5},
            revision_log=[{"iteration": 1}],
            summative_evaluation={"effectiveness_score": 8.0},
        )

        addie = map_to_addie_output(state)

        # Analysis 매핑
        assert addie["analysis"]["instructional_goal"]["goal_statement"] == "테스트 목표"
        assert addie["analysis"]["learner_analysis"]["target_audience"] == "신입사원"
        assert addie["analysis"]["task_analysis"]["task_type"] == "조합형"

        # Design 매핑
        assert addie["design"]["learning_objectives"]["terminal_objective"]["id"] == "PO-T01"

        # Development 매핑
        assert len(addie["development"]["materials"]) == 1

        # Implementation 매핑
        assert addie["implementation"]["formative_evaluation"]["quality_score"] == 7.5
        assert len(addie["implementation"]["revision_history"]) == 1

        # Evaluation 매핑
        assert addie["evaluation"]["summative_evaluation"]["effectiveness_score"] == 8.0


class TestDickCareyPhases:
    """Dick & Carey 단계 테스트"""

    def test_phase_values(self):
        """단계 값 테스트"""
        from dick_carey_agent.state import DickCareyPhase

        valid_phases = [
            "goal",
            "instructional_analysis",
            "learner_context",
            "performance_objectives",
            "assessment_instruments",
            "instructional_strategy",
            "instructional_materials",
            "formative_evaluation",
            "revision",
            "summative_evaluation",
            "complete",
        ]

        # 모든 단계가 유효한지 확인
        for phase in valid_phases:
            state = create_initial_state(ScenarioInput(scenario_id="TEST"))
            state["current_phase"] = phase
            assert state["current_phase"] == phase
