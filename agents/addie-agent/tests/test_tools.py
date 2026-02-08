"""도구 폴백 함수 테스트

LLM 호출 없이 폴백 함수만 테스트합니다.
"""

import pytest
from addie_agent.tools.analysis import (
    _fallback_analyze_learner,
    _fallback_analyze_context,
    _fallback_analyze_task,
    _fallback_analyze_needs,
)
from addie_agent.tools.design import (
    _fallback_design_objectives,
    _fallback_design_assessment,
    _fallback_design_strategy,
)
from addie_agent.tools.development import (
    _fallback_create_lesson_plan,
    _fallback_create_materials,
)
from addie_agent.tools.implementation import (
    _fallback_create_implementation_plan,
)
from addie_agent.tools.evaluation import (
    _fallback_create_quiz_items,
    _fallback_create_rubric,
    _fallback_create_program_evaluation,
)


class TestAnalysisTools:
    """Analysis 도구 테스트"""

    def test_analyze_learner_newcomer(self):
        """신입사원 학습자 분석 테스트"""
        result = _fallback_analyze_learner("신입사원", "기초", None)

        assert result["target_audience"] == "신입사원"
        assert len(result["characteristics"]) >= 5
        assert len(result["learning_preferences"]) >= 4
        assert len(result["challenges"]) >= 3
        assert result["motivation"] is not None

    def test_analyze_learner_elementary(self):
        """초등학생 학습자 분석 테스트"""
        result = _fallback_analyze_learner("초등학교 5학년", None, None)

        assert "초등" not in result["target_audience"] or len(result["characteristics"]) >= 5
        assert "놀이" in " ".join(result["learning_preferences"]) or len(result["learning_preferences"]) >= 4

    def test_analyze_context_online(self):
        """온라인 환경 분석 테스트"""
        result = _fallback_analyze_context("온라인", "2시간", 30, None, None)

        assert result["environment"] == "온라인"
        assert len(result["constraints"]) >= 3
        assert len(result["resources"]) >= 3
        assert len(result["technical_requirements"]) >= 2
        assert "인터넷" in " ".join(result["technical_requirements"])

    def test_analyze_context_offline(self):
        """대면 환경 분석 테스트"""
        result = _fallback_analyze_context("대면 교실", "3시간", 20, None, None)

        assert len(result["constraints"]) >= 3
        assert len(result["technical_requirements"]) >= 2

    def test_analyze_task(self):
        """과제 분석 테스트"""
        learning_goals = ["조직 문화 이해", "업무 프로세스 파악", "협업 도구 활용"]
        result = _fallback_analyze_task(learning_goals, "비즈니스", "medium")

        assert len(result["main_topics"]) >= 3
        assert len(result["subtopics"]) >= 6
        assert len(result["prerequisites"]) >= 2

    def test_analyze_task_includes_review_summary(self):
        """과제 분석에 review_summary 포함 테스트 (A-10)"""
        learning_goals = ["Python 프로그래밍", "데이터 분석"]
        result = _fallback_analyze_task(learning_goals, "IT", "medium")

        assert "review_summary" in result
        assert isinstance(result["review_summary"], str)
        assert len(result["review_summary"]) > 50  # 의미있는 요약

    def test_analyze_needs_returns_valid_schema(self):
        """요구분석 스키마 검증 테스트"""
        learning_goals = ["조직 문화 이해", "업무 프로세스 파악"]
        result = _fallback_analyze_needs(learning_goals)

        assert "gap_analysis" in result
        assert "root_causes" in result
        assert "training_needs" in result
        assert "non_training_solutions" in result
        assert "priority" in result
        assert "recommendation" in result
        assert isinstance(result["non_training_solutions"], list)

    def test_analyze_needs_includes_priority_matrix(self):
        """요구분석에 priority_matrix 포함 테스트 (A-4 확장)"""
        learning_goals = ["목표1", "목표2", "목표3"]
        result = _fallback_analyze_needs(learning_goals)

        assert "priority_matrix" in result
        matrix = result["priority_matrix"]
        assert "high_urgency_high_impact" in matrix
        assert "high_urgency_low_impact" in matrix
        assert "low_urgency_high_impact" in matrix
        assert "low_urgency_low_impact" in matrix


class TestDesignTools:
    """Design 도구 테스트"""

    def test_design_objectives(self):
        """학습 목표 설계 테스트"""
        learning_goals = ["목표1", "목표2", "목표3"]
        result = _fallback_design_objectives(learning_goals, "신입사원", "medium")

        assert len(result) >= 5
        for obj in result:
            assert "id" in obj
            assert "level" in obj
            assert "statement" in obj
            assert "bloom_verb" in obj

    def test_design_assessment(self):
        """평가 계획 테스트"""
        objectives = [{"id": "LO-001", "level": "이해"}]
        result = _fallback_design_assessment(objectives, "2시간", "온라인")

        assert len(result["diagnostic"]) >= 2
        assert len(result["formative"]) >= 2
        assert len(result["summative"]) >= 2

    def test_design_strategy(self):
        """교수 전략 테스트"""
        result = _fallback_design_strategy(
            ["주제1", "주제2"],
            "신입사원",
            "2시간",
            "대면 교육",
        )

        assert result["model"] == "Gagné's 9 Events"
        assert len(result["sequence"]) == 9
        assert len(result["methods"]) >= 3


class TestDevelopmentTools:
    """Development 도구 테스트"""

    def test_create_lesson_plan(self):
        """레슨 플랜 생성 테스트"""
        objectives = [{"id": f"LO-{i:03d}"} for i in range(1, 6)]
        strategy = {"model": "Gagné's 9 Events", "sequence": []}
        result = _fallback_create_lesson_plan(
            objectives, strategy, "2시간", ["주제1", "주제2", "주제3"]
        )

        assert len(result["modules"]) >= 3
        for module in result["modules"]:
            assert len(module["activities"]) >= 3

    def test_create_materials(self):
        """학습 자료 생성 테스트"""
        lesson_plan = {"modules": [{"title": "모듈1"}]}
        result = _fallback_create_materials(lesson_plan, "대면", "신입사원")

        assert len(result) >= 5

        # PPT 자료에 slide_contents 확인
        ppt_materials = [m for m in result if m.get("type") == "PPT"]
        if ppt_materials:
            assert "slide_contents" in ppt_materials[0]

    def test_create_materials_includes_storyboard(self):
        """학습 자료에 storyboard 포함 테스트 (D-18)"""
        lesson_plan = {"modules": [{"title": "모듈1"}, {"title": "모듈2"}]}
        result = _fallback_create_materials(lesson_plan, "온라인", "직장인")

        # PPT 자료에 storyboard 확인
        ppt_materials = [m for m in result if m.get("type") == "PPT"]
        assert len(ppt_materials) > 0
        assert "storyboard" in ppt_materials[0]
        storyboard = ppt_materials[0]["storyboard"]
        assert len(storyboard) >= 2  # 최소 2개 프레임

        # storyboard 프레임 구조 검증
        for frame in storyboard:
            assert "frame_number" in frame
            assert "screen_title" in frame
            assert "visual_description" in frame


class TestImplementationTools:
    """Implementation 도구 테스트"""

    def test_create_implementation_plan_online(self):
        """온라인 실행 계획 테스트"""
        result = _fallback_create_implementation_plan(
            {"modules": []}, "온라인", "직장인", 30
        )

        assert len(result["facilitator_guide"]) >= 200
        assert len(result["learner_guide"]) >= 200
        assert len(result["technical_requirements"]) >= 2
        assert result["support_plan"] is not None

    def test_create_implementation_plan_offline(self):
        """대면 실행 계획 테스트"""
        result = _fallback_create_implementation_plan(
            {"modules": []}, "대면 교실", "신입사원", 20
        )

        assert len(result["facilitator_guide"]) >= 200
        assert len(result["learner_guide"]) >= 200

    def test_create_implementation_plan_includes_operator_guide(self):
        """실행 계획에 operator_guide 포함 테스트 (Dev-21)"""
        result = _fallback_create_implementation_plan(
            {"modules": []}, "온라인", "직장인", 30
        )

        assert "operator_guide" in result
        assert isinstance(result["operator_guide"], str)
        assert len(result["operator_guide"]) >= 200

    def test_create_implementation_plan_includes_orientation_plan(self):
        """실행 계획에 orientation_plan 포함 테스트 (I-24)"""
        result = _fallback_create_implementation_plan(
            {"modules": []}, "온라인", "직장인", 30
        )

        assert "orientation_plan" in result
        assert isinstance(result["orientation_plan"], str)
        assert len(result["orientation_plan"]) >= 200

    def test_create_implementation_plan_includes_pilot_plan(self):
        """실행 계획에 pilot_plan 포함 테스트 (I-26)"""
        result = _fallback_create_implementation_plan(
            {"modules": []}, "온라인", "직장인", 30
        )

        assert "pilot_plan" in result
        pilot = result["pilot_plan"]
        assert "pilot_scope" in pilot
        assert "participants" in pilot
        assert "duration" in pilot
        assert "success_criteria" in pilot
        assert "data_collection" in pilot
        assert "contingency_plan" in pilot
        assert len(pilot["success_criteria"]) >= 3
        assert len(pilot["data_collection"]) >= 3


class TestEvaluationTools:
    """Evaluation 도구 테스트"""

    def test_create_quiz_items(self):
        """퀴즈 문항 생성 테스트"""
        objectives = [{"id": f"LO-{i:03d}"} for i in range(1, 6)]
        result = _fallback_create_quiz_items(
            objectives, ["주제1", "주제2"], "medium", 10
        )

        assert len(result) >= 10
        for item in result:
            assert "id" in item
            assert "question" in item
            assert "options" in item
            assert "answer" in item
            assert "explanation" in item
            assert "difficulty" in item

        # 난이도 분산 확인
        difficulties = [item["difficulty"] for item in result]
        assert "쉬움" in difficulties
        assert "보통" in difficulties
        assert "어려움" in difficulties

    def test_create_rubric(self):
        """평가 루브릭 생성 테스트"""
        objectives = [{"id": "LO-001"}]
        result = _fallback_create_rubric(objectives, "종합 평가")

        assert len(result["criteria"]) >= 5
        assert "excellent" in result["levels"]
        assert "good" in result["levels"]
        assert "needs_improvement" in result["levels"]

    def test_create_program_evaluation(self):
        """성과평가 계획 테스트"""
        objectives = [{"id": f"LO-{i:03d}"} for i in range(1, 6)]
        result = _fallback_create_program_evaluation(
            "신입사원 온보딩 교육",
            objectives,
            "신입사원"
        )

        assert "program_title" in result
        assert "evaluation_model" in result
        assert result["evaluation_model"] == "Kirkpatrick 4-Level"
        assert "levels" in result
        assert "roi_calculation" in result
        assert "evaluation_schedule" in result
        assert "success_criteria" in result

    def test_create_program_evaluation_includes_adoption_decision(self):
        """성과평가에 adoption_decision 포함 테스트 (E-32)"""
        objectives = [{"id": f"LO-{i:03d}"} for i in range(1, 6)]
        result = _fallback_create_program_evaluation(
            "디지털 역량 강화 교육",
            objectives,
            "직장인"
        )

        assert "adoption_decision" in result
        decision = result["adoption_decision"]
        assert "recommendation" in decision
        assert decision["recommendation"] in ["adopt", "modify", "reject"]
        assert "rationale" in decision
        assert "conditions" in decision
        assert "next_steps" in decision
        assert len(decision["conditions"]) >= 3
        assert len(decision["next_steps"]) >= 3
