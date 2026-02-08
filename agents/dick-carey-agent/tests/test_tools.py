"""
Tools 테스트

Dick & Carey 11개 도구의 폴백 함수 테스트
"""

import pytest
from dick_carey_agent.tools.goal_analysis import (
    _fallback_set_instructional_goal,
    _fallback_analyze_instruction,
    _fallback_analyze_entry_behaviors,
    _fallback_analyze_context,
)
from dick_carey_agent.tools.objective_assessment import (
    _fallback_write_performance_objectives,
    _fallback_develop_assessment_instruments,
)
from dick_carey_agent.tools.strategy_materials import (
    _fallback_develop_instructional_strategy,
    _fallback_develop_instructional_materials,
)
from dick_carey_agent.tools.evaluation import (
    _fallback_conduct_formative_evaluation,
    _fallback_revise_instruction,
    _fallback_conduct_summative_evaluation,
)


class TestGoalAnalysisTools:
    """1-3단계 도구 폴백 테스트"""

    def test_set_instructional_goal_fallback(self):
        """교수목적 설정 폴백 테스트"""
        result = _fallback_set_instructional_goal(
            learning_goals=["목표 1", "목표 2"],
            target_audience="신입사원",
            current_state="기초 수준",
            desired_state="전문가 수준",
        )

        assert "goal_statement" in result
        assert "target_domain" in result
        assert "performance_gap" in result
        assert "신입사원" in result["goal_statement"]

    def test_analyze_instruction_fallback(self):
        """교수분석 폴백 테스트"""
        result = _fallback_analyze_instruction(
            instructional_goal="테스트 목표",
            domain="IT",
            learning_goals=["목표 1", "목표 2"],
        )

        assert "task_type" in result
        assert "sub_skills" in result
        assert "skill_hierarchy" in result
        assert "entry_skills" in result
        assert len(result["sub_skills"]) >= 5

    def test_analyze_entry_behaviors_fallback(self):
        """학습자 분석 폴백 테스트"""
        result = _fallback_analyze_entry_behaviors(
            target_audience="신입사원",
            prior_knowledge="기초 지식",
            entry_skills=["기초 용어", "컴퓨터 활용"],
        )

        assert "target_audience" in result
        assert "entry_behaviors" in result
        assert "characteristics" in result
        assert "learning_preferences" in result
        assert "motivation" in result
        assert len(result["characteristics"]) >= 5

    def test_analyze_context_fallback_online(self):
        """환경 분석 폴백 테스트 (온라인)"""
        result = _fallback_analyze_context(
            learning_environment="온라인",
            duration="2시간",
            performance_context="실무 현장",
            class_size=30,
            resources=None,
        )

        assert "performance_context" in result
        assert "learning_context" in result
        assert "constraints" in result
        assert "resources" in result
        assert len(result["constraints"]) >= 3

    def test_analyze_context_fallback_offline(self):
        """환경 분석 폴백 테스트 (대면)"""
        result = _fallback_analyze_context(
            learning_environment="대면 교실",
            duration="4시간",
            performance_context=None,
            class_size=40,
            resources=None,
        )

        assert "대규모 그룹 관리 필요" in result["constraints"]


class TestObjectiveAssessmentTools:
    """4-5단계 도구 폴백 테스트"""

    def test_write_performance_objectives_fallback(self):
        """수행목표 진술 폴백 테스트"""
        sub_skills = [
            {"skill_name": "핵심 개념 이해", "description": "기초 개념 이해"},
            {"skill_name": "사례 분석", "description": "사례 분석"},
            {"skill_name": "문제 진단", "description": "문제 진단"},
            {"skill_name": "해결책 설계", "description": "해결책 설계"},
            {"skill_name": "결과 평가", "description": "결과 평가"},
        ]

        result = _fallback_write_performance_objectives(
            instructional_goal="테스트 목표",
            sub_skills=sub_skills,
            target_audience="신입사원",
        )

        assert "terminal_objective" in result
        assert "enabling_objectives" in result
        assert len(result["enabling_objectives"]) >= 5
        assert "objective_name" in result["terminal_objective"]
        assert "audience" in result["terminal_objective"]
        assert "behavior" in result["terminal_objective"]
        assert "condition" in result["terminal_objective"]
        assert "degree" in result["terminal_objective"]

    def test_develop_assessment_instruments_fallback(self):
        """평가도구 개발 폴백 테스트"""
        objectives = {
            "terminal_objective": {"objective_name": "종합 수행 목표", "behavior": "종합적으로 수행"},
            "enabling_objectives": [
                {"objective_name": "개념 이해 목표", "behavior": "개념 이해"},
                {"objective_name": "사례 분석 목표", "behavior": "사례 분석"},
                {"objective_name": "문제 해결 목표", "behavior": "문제 해결"},
            ],
        }

        result = _fallback_develop_assessment_instruments(
            performance_objectives=objectives,
            learning_environment="온라인",
            duration="2시간",
        )

        assert "entry_test" in result
        assert "practice_tests" in result
        assert "post_test" in result
        assert "alignment_matrix" in result
        assert len(result["entry_test"]) >= 3
        assert len(result["practice_tests"]) >= 3
        assert len(result["post_test"]) >= 5


class TestStrategyMaterialsTools:
    """6-7단계 도구 폴백 테스트"""

    def test_develop_instructional_strategy_fallback(self):
        """교수전략 개발 폴백 테스트"""
        result = _fallback_develop_instructional_strategy(
            performance_objectives={},
            learner_analysis={},
            learning_environment="온라인",
            duration="2시간",
        )

        assert "pre_instructional" in result
        assert "content_presentation" in result
        assert "learner_participation" in result
        assert "assessment" in result
        assert "delivery_method" in result
        assert "grouping_strategy" in result
        assert len(result["learner_participation"]["practice_activities"]) >= 3

    def test_develop_instructional_materials_fallback(self):
        """교수자료 개발 폴백 테스트"""
        result = _fallback_develop_instructional_materials(
            instructional_strategy={},
            performance_objectives={},
            learning_environment="온라인",
            duration="2시간",
            topic_title="테스트 주제",
        )

        assert "instructor_guide" in result
        assert "learner_materials" in result
        assert "media_list" in result
        assert "slide_contents" in result
        assert len(result["learner_materials"]) >= 3
        assert len(result["slide_contents"]) >= 10


class TestEvaluationTools:
    """8-10단계 도구 폴백 테스트"""

    def test_conduct_formative_evaluation_fallback_first_iteration(self):
        """형성평가 폴백 테스트 (1차)"""
        result = _fallback_conduct_formative_evaluation(
            instructional_materials={},
            performance_objectives={},
            assessment_instruments={},
            iteration=1,
        )

        assert "quality_score" in result
        assert "one_to_one_findings" in result
        assert "small_group_findings" in result
        assert "field_trial_findings" in result
        assert "strengths" in result
        assert "weaknesses" in result
        assert "revision_recommendations" in result
        assert result["quality_score"] == 6.5  # 1차 기본 점수

    def test_conduct_formative_evaluation_fallback_improvement(self):
        """형성평가 폴백 테스트 (반복에 따른 점수 향상)"""
        result_1 = _fallback_conduct_formative_evaluation({}, {}, {}, iteration=1)
        result_2 = _fallback_conduct_formative_evaluation({}, {}, {}, iteration=2)
        result_3 = _fallback_conduct_formative_evaluation({}, {}, {}, iteration=3)

        # 반복에 따라 점수 향상
        assert result_2["quality_score"] > result_1["quality_score"]
        assert result_3["quality_score"] > result_2["quality_score"]

    def test_revise_instruction_fallback(self):
        """교수프로그램 수정 폴백 테스트"""
        formative = {
            "revision_recommendations": [
                "개념 설명 보강",
                "활동 시간 조정",
                "보충 자료 개발",
            ]
        }

        result = _fallback_revise_instruction(
            formative_evaluation=formative,
            current_state={},
            iteration=1,
        )

        assert "iteration" in result
        assert "revision_items" in result
        assert "summary" in result
        assert result["iteration"] == 1
        assert len(result["revision_items"]) >= 3

    def test_conduct_summative_evaluation_fallback(self):
        """총괄평가 폴백 테스트"""
        result = _fallback_conduct_summative_evaluation(
            final_state={},
            performance_objectives={},
            total_iterations=2,
        )

        assert "effectiveness_score" in result
        assert "efficiency_analysis" in result
        assert "learner_satisfaction" in result
        assert "goal_achievement" in result
        assert "recommendations" in result
        assert "decision" in result
        assert result["effectiveness_score"] >= 7.0  # 기준 충족


class TestFeedbackLoop:
    """피드백 루프 관련 테스트"""

    def test_quality_threshold_check(self):
        """품질 기준 체크 테스트"""
        # 낮은 점수 - 수정 필요
        result_low = _fallback_conduct_formative_evaluation({}, {}, {}, iteration=1)
        assert result_low["quality_score"] < 7.0

        # 3차 반복 - 기준 근접 또는 충족
        result_high = _fallback_conduct_formative_evaluation({}, {}, {}, iteration=3)
        assert result_high["quality_score"] >= 7.0 or result_high["quality_score"] > result_low["quality_score"]

    def test_revision_items_have_required_fields(self):
        """수정 항목 필수 필드 테스트"""
        result = _fallback_revise_instruction(
            formative_evaluation={"revision_recommendations": ["수정 1", "수정 2", "수정 3"]},
            current_state={},
            iteration=1,
        )

        for item in result["revision_items"]:
            assert "issue" in item
            assert "target_phase" in item
            assert "action" in item
            assert "status" in item
