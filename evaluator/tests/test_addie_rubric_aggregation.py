"""
ADDIE 루브릭 점수 통합 로직 단위 테스트

테스트 대상:
- ADDIERubricEvaluator._aggregate_sub_scores_to_item()
- ADDIERubricEvaluator._build_final_score_from_sub_items()
"""

import pytest
from unittest.mock import MagicMock

from isd_evaluator.models import ADDIEPhase, ADDIEScore, PhaseScore


class TestAggregateSubScoresToItem:
    """_aggregate_sub_scores_to_item 메서드 테스트

    33개 소단계 점수를 13개 항목으로 집계 (평균 계산)
    """

    def test_aggregate_a1_items(self, mock_evaluator, item_mapping):
        """A1 항목 집계 테스트 (sub_items: 1, 2, 3, 4 - 요구분석)"""
        all_sub_scores = {
            1: 8.0,   # 문제 확인 및 정의
            2: 7.0,   # 차이분석
            3: 6.0,   # 수행분석
            4: 9.0,   # 요구 우선순위 결정
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("A1", all_sub_scores)

        # 평균: (8 + 7 + 6 + 9) / 4 = 7.5
        assert result == 7.5

    def test_aggregate_a2_multiple_items(self, mock_evaluator):
        """A2 항목 집계 테스트 (sub_items: 5, 6 - 학습자 및 환경분석)"""
        all_sub_scores = {
            5: 8.0,   # 학습자 분석
            6: 9.0,   # 환경 분석
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("A2", all_sub_scores)

        # 평균: (8.0 + 9.0) / 2 = 8.5
        assert result == 8.5

    def test_aggregate_a3_multiple_items(self, mock_evaluator):
        """A3 항목 집계 테스트 (sub_items: 7, 8, 9, 10 - 과제 및 목표분석)"""
        all_sub_scores = {
            7: 8.5,   # 초기 학습목표 분석
            8: 7.0,   # 하위 기능 분석
            9: 8.0,   # 출발점 행동 분석
            10: 6.5,  # 과제분석 결과 검토·정리
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("A3", all_sub_scores)

        # 평균: (8.5 + 7.0 + 8.0 + 6.5) / 4 = 7.5
        assert result == 7.5

    def test_aggregate_d1_multiple_items(self, mock_evaluator):
        """D1 항목 집계 테스트 (sub_items: 11, 12 - 평가 및 목표 정렬 설계)"""
        all_sub_scores = {
            11: 8.0,  # 학습목표 정교화
            12: 7.0,  # 평가 계획 수립
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("D1", all_sub_scores)

        # 평균: (8.0 + 7.0) / 2 = 7.5
        assert result == 7.5

    def test_aggregate_d2_multiple_items(self, mock_evaluator):
        """D2 항목 집계 테스트 (sub_items: 13, 14, 15, 16, 17 - 교수전략 및 학습경험 설계)"""
        all_sub_scores = {
            13: 7.0,  # 교수 내용 선정
            14: 8.0,  # 교수적 전략 수립
            15: 6.0,  # 비교수적 전략 수립
            16: 7.5,  # 매체 선정과 활용 계획
            17: 9.0,  # 학습활동 및 시간 구조화
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("D2", all_sub_scores)

        # 평균: (7.0 + 8.0 + 6.0 + 7.5 + 9.0) / 5 = 7.5
        assert result == 7.5

    def test_aggregate_d3_single_item(self, mock_evaluator):
        """D3 항목 집계 테스트 (단일 항목: 18 - 프로토타입 구조 설계)"""
        all_sub_scores = {18: 7.5}  # 스토리보드/화면 흐름 설계

        result = mock_evaluator._aggregate_sub_scores_to_item("D3", all_sub_scores)

        # 단일 항목이므로 그대로 반환
        assert result == 7.5

    def test_aggregate_dev1_multiple_items(self, mock_evaluator):
        """Dev1 항목 집계 테스트 (sub_items: 19, 20, 21, 22 - 프로토타입 개발)"""
        all_sub_scores = {
            19: 8.0,  # 학습자용 자료 개발
            20: 7.0,  # 교수자용 매뉴얼 개발
            21: 6.0,  # 운영자용 매뉴얼 개발
            22: 9.0,  # 평가 도구·문항 개발
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("Dev1", all_sub_scores)

        # 평균: (8.0 + 7.0 + 6.0 + 9.0) / 4 = 7.5
        assert result == 7.5

    def test_aggregate_dev2_single_item(self, mock_evaluator):
        """Dev2 항목 집계 테스트 (단일 항목: 23 - 개발 결과 검토 및 수정)"""
        all_sub_scores = {23: 7.5}  # 전문가 검토

        result = mock_evaluator._aggregate_sub_scores_to_item("Dev2", all_sub_scores)

        # 단일 항목이므로 그대로 반환
        assert result == 7.5

    def test_aggregate_i1_multiple_items(self, mock_evaluator):
        """I1 항목 집계 테스트 (sub_items: 24, 25 - 프로그램 실행 준비)"""
        all_sub_scores = {
            24: 8.0,  # 교수자·운영자 오리엔테이션
            25: 7.0,  # 시스템/환경 점검
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("I1", all_sub_scores)

        # 평균: (8.0 + 7.0) / 2 = 7.5
        assert result == 7.5

    def test_aggregate_i2_multiple_items(self, mock_evaluator):
        """I2 항목 집계 테스트 (sub_items: 26, 27 - 프로그램 실행)"""
        all_sub_scores = {
            26: 8.0,  # 프로토타입 실행
            27: 7.0,  # 운영 모니터링 및 지원
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("I2", all_sub_scores)

        # 평균: (8.0 + 7.0) / 2 = 7.5
        assert result == 7.5

    def test_aggregate_e1_multiple_items(self, mock_evaluator):
        """E1 항목 집계 테스트 (sub_items: 28, 29 - 형성평가)"""
        all_sub_scores = {
            28: 8.0,  # 파일럿/초기 실행 중 자료 수집
            29: 7.0,  # 형성평가 결과 기반 1차 프로그램 개선
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("E1", all_sub_scores)

        # 평균: (8.0 + 7.0) / 2 = 7.5
        assert result == 7.5

    def test_aggregate_e2_multiple_items(self, mock_evaluator):
        """E2 항목 집계 테스트 (sub_items: 30, 31, 32 - 총괄평가 및 채택 결정)"""
        all_sub_scores = {
            30: 8.0,  # 총괄 평가 문항 개발(검사도구 완성)
            31: 7.0,  # 총괄평가 시행 및 프로그램 효과 분석
            32: 6.5,  # 프로그램 채택 여부 결정
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("E2", all_sub_scores)

        # 평균: (8.0 + 7.0 + 6.5) / 3 = 7.16...
        assert round(result, 2) == 7.17

    def test_aggregate_e3_single_item(self, mock_evaluator):
        """E3 항목 집계 테스트 (단일 항목: 33 - 프로그램 개선 및 환류)"""
        all_sub_scores = {33: 7.5}  # 프로그램 개선

        result = mock_evaluator._aggregate_sub_scores_to_item("E3", all_sub_scores)

        # 단일 항목이므로 그대로 반환
        assert result == 7.5

    def test_aggregate_missing_sub_scores_uses_default(self, mock_evaluator):
        """누락된 sub_score는 기본값 5.0 사용"""
        all_sub_scores = {
            1: 8.0,
            # 2, 3, 4 누락
        }

        result = mock_evaluator._aggregate_sub_scores_to_item("A1", all_sub_scores)

        # A1 = [1, 2, 3, 4] (요구분석)
        # (8 + 5 + 5 + 5) / 4 = 5.75
        assert result == 5.75

    def test_aggregate_unknown_item_returns_default(self, mock_evaluator):
        """알 수 없는 항목 ID는 기본값 5.0 반환"""
        all_sub_scores = {1: 8.0, 2: 7.0}

        result = mock_evaluator._aggregate_sub_scores_to_item("UNKNOWN", all_sub_scores)

        assert result == 5.0

    def test_aggregate_empty_sub_scores(self, mock_evaluator):
        """빈 sub_scores로 집계 시 기본값 반환"""
        all_sub_scores = {}

        result = mock_evaluator._aggregate_sub_scores_to_item("A1", all_sub_scores)

        # 모든 항목이 기본값 5.0
        assert result == 5.0

    def test_aggregate_all_13_items(self, mock_evaluator, complete_sub_scores, item_mapping):
        """13개 항목 모두 집계 테스트"""
        for item_id in item_mapping.keys():
            result = mock_evaluator._aggregate_sub_scores_to_item(item_id, complete_sub_scores)
            # 모든 점수가 7.5이므로 평균도 7.5
            assert result == 7.5


class TestBuildFinalScoreFromSubItems:
    """_build_final_score_from_sub_items 메서드 테스트"""

    def test_build_final_score_structure(
        self, mock_evaluator, complete_sub_scores, complete_sub_reasoning
    ):
        """ADDIEScore 구조 확인"""
        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            ["누락 항목 1"],
            ["약점 영역 1"],
        )

        assert isinstance(result, ADDIEScore)
        assert hasattr(result, 'analysis')
        assert hasattr(result, 'design')
        assert hasattr(result, 'development')
        assert hasattr(result, 'implementation')
        assert hasattr(result, 'evaluation')

    def test_build_final_score_phase_item_counts(
        self, mock_evaluator, complete_sub_scores, complete_sub_reasoning
    ):
        """각 단계별 항목 수 확인 (CSV 중단계 기준)"""
        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            [], [],
        )

        # Analysis: A1, A2, A3 (3개 중단계)
        assert len(result.analysis.items) == 3

        # Design: D1, D2, D3 (3개 중단계)
        assert len(result.design.items) == 3

        # Development: Dev1, Dev2 (2개 중단계 - CSV 기준)
        assert len(result.development.items) == 2

        # Implementation: I1, I2 (2개 중단계)
        assert len(result.implementation.items) == 2

        # Evaluation: E1, E2, E3 (3개 중단계 - CSV 기준)
        assert len(result.evaluation.items) == 3

    def test_build_final_score_normalized(
        self, mock_evaluator, complete_sub_scores, complete_sub_reasoning
    ):
        """정규화 점수 확인 (0-100)"""
        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            [], [],
        )

        # 정규화 점수 범위 확인
        assert 0 <= result.normalized_score <= 100

        # 모든 점수가 7.5일 때 정규화 점수
        # 7.5 / 10 * 100 = 75점 근처
        assert 74 <= result.normalized_score <= 76

    def test_build_final_score_with_varying_scores(
        self, mock_evaluator, varying_sub_scores, complete_sub_reasoning
    ):
        """다양한 점수로 테스트"""
        result = mock_evaluator._build_final_score_from_sub_items(
            varying_sub_scores,
            complete_sub_reasoning,
            [], []
        )

        # Analysis 점수가 가장 높음 (9.0)
        # Development 점수가 가장 낮음 (5.0)
        assert result.analysis.average_score > result.development.average_score

    def test_build_final_score_phase_percentages(
        self, mock_evaluator, complete_sub_scores, complete_sub_reasoning
    ):
        """각 단계별 백분율 확인"""
        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            [], [],
        )

        # 모든 점수가 7.5이므로 각 단계의 백분율도 75%
        assert 74 <= result.analysis.percentage <= 76
        assert 74 <= result.design.percentage <= 76
        assert 74 <= result.development.percentage <= 76
        assert 74 <= result.implementation.percentage <= 76
        assert 74 <= result.evaluation.percentage <= 76

    def test_build_final_score_total_raw(
        self, mock_evaluator, complete_sub_scores, complete_sub_reasoning
    ):
        """원점수 총합 확인"""
        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            [], [],
        )

        # 13개 항목 × 7.5점 = 97.5점
        # (단, 각 항목은 소단계 평균이므로 정확히 97.5가 아닐 수 있음)
        assert result.total_raw > 0

    def test_build_final_score_improvements(
        self, mock_evaluator, complete_sub_scores, complete_sub_reasoning
    ):
        """improvements 리스트 확인"""
        missing = ["요소1", "요소2", "요소3", "요소4", "요소5", "요소6"]
        weak = ["약점1", "약점2", "약점3", "약점4", "약점5", "약점6"]

        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            missing,
            weak
        )

        # improvements는 missing[:5] + weak[:5] = 최대 10개
        assert len(result.improvements) <= 10

    def test_build_final_score_overall_assessment(
        self, mock_evaluator, complete_sub_scores, complete_sub_reasoning
    ):
        """종합 평가 메시지 확인"""
        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            [], [],
        )

        # overall_assessment에 "33개 소단계" 언급
        assert "33개 소단계" in result.overall_assessment or "소단계" in result.overall_assessment


class TestBuildFinalScoreEdgeCases:
    """_build_final_score_from_sub_items 경계 케이스"""

    def test_empty_sub_scores(self, mock_evaluator):
        """빈 sub_scores 처리"""
        result = mock_evaluator._build_final_score_from_sub_items({}, {}, [], [])

        # 기본값 5.0으로 계산되어야 함
        assert isinstance(result, ADDIEScore)
        assert result.normalized_score > 0

    def test_partial_sub_scores(self, mock_evaluator):
        """일부만 있는 sub_scores"""
        sub_scores = {1: 8.0, 2: 7.0, 3: 9.0}  # Analysis 일부만
        sub_reasoning = {1: "평가1", 2: "평가2", 3: "평가3"}

        result = mock_evaluator._build_final_score_from_sub_items(
            sub_scores, sub_reasoning, [], []
        )

        # 결과가 생성되어야 함
        assert isinstance(result, ADDIEScore)

    def test_all_max_scores(self, mock_evaluator):
        """모든 점수가 최대값(10.0)일 때"""
        sub_scores = {i: 10.0 for i in range(1, 34)}
        sub_reasoning = {i: f"만점 {i}" for i in range(1, 34)}

        result = mock_evaluator._build_final_score_from_sub_items(
            sub_scores, sub_reasoning, [], []
        )

        # 정규화 점수가 100에 가까워야 함
        assert result.normalized_score >= 99

    def test_all_min_scores(self, mock_evaluator):
        """모든 점수가 최소값(0.0)일 때"""
        sub_scores = {i: 0.0 for i in range(1, 34)}
        sub_reasoning = {i: f"최저 {i}" for i in range(1, 34)}

        result = mock_evaluator._build_final_score_from_sub_items(
            sub_scores, sub_reasoning, [], []
        )

        # 정규화 점수가 0에 가까워야 함
        assert result.normalized_score <= 1

    def test_phase_score_types(self, mock_evaluator, complete_sub_scores, complete_sub_reasoning):
        """각 PhaseScore의 타입 확인"""
        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            [], [],
        )

        assert isinstance(result.analysis, PhaseScore)
        assert isinstance(result.design, PhaseScore)
        assert isinstance(result.development, PhaseScore)
        assert isinstance(result.implementation, PhaseScore)
        assert isinstance(result.evaluation, PhaseScore)

    def test_rubric_item_fields(self, mock_evaluator, complete_sub_scores, complete_sub_reasoning):
        """RubricItem 필드 확인"""
        result = mock_evaluator._build_final_score_from_sub_items(
            complete_sub_scores,
            complete_sub_reasoning,
            [], [],
        )

        # 첫 번째 Analysis 항목 확인
        first_item = result.analysis.items[0]
        assert hasattr(first_item, 'item_id')
        assert hasattr(first_item, 'phase')
        assert hasattr(first_item, 'name')
        assert hasattr(first_item, 'score')
        assert hasattr(first_item, 'reasoning')
