"""ContextMatrix 단위 테스트"""

import pytest
from pathlib import Path
from shared.models.context_matrix import (
    ContextMatrix,
    ContextCombination,
    ContextItem,
    SUB_DIMENSION_TO_FIELD,
)


@pytest.fixture
def csv_path() -> str:
    """테스트용 CSV 경로"""
    current_dir = Path(__file__).parent
    return str(current_dir.parent.parent / "참고문서 " / "Context Matrix - 컨텍스트 축.csv")


@pytest.fixture
def context_matrix(csv_path: str) -> ContextMatrix:
    """ContextMatrix 인스턴스"""
    return ContextMatrix(csv_path)


class TestContextMatrixLoad:
    """CSV 로드 테스트"""

    def test_load_csv(self, context_matrix: ContextMatrix):
        """CSV 파일 로드 검증"""
        assert len(context_matrix.items) == 51
        assert len(context_matrix.dimensions) == 5

    def test_dimensions(self, context_matrix: ContextMatrix):
        """5개 대단계 존재 확인"""
        expected = ["학습자 특성", "기관 맥락", "교육 도메인", "전달 방식", "제약 조건"]
        assert set(context_matrix.get_dimensions()) == set(expected)

    def test_sub_dimensions(self, context_matrix: ContextMatrix):
        """중단계 구조 확인"""
        # 학습자 특성: 연령, 학력수준, 도메인지식수준, 직업·역할
        learner_subs = context_matrix.get_sub_dimensions("학습자 특성")
        assert len(learner_subs) == 4
        assert "연령" in learner_subs
        assert "학력수준" in learner_subs

    def test_options(self, context_matrix: ContextMatrix):
        """소단계(옵션) 확인"""
        age_options = context_matrix.get_options("학습자 특성", "연령")
        assert len(age_options) == 4
        assert "10대" in age_options
        assert "40대 이상" in age_options

    def test_summary(self, context_matrix: ContextMatrix):
        """요약 정보 확인"""
        summary = context_matrix.summary()
        assert summary["total_items"] == 51
        assert "dimensions" in summary
        assert summary["total_combinations"] > 1000  # 조합 수가 많음


class TestContextCombination:
    """ContextCombination 테스트"""

    def test_sample_combination(self, context_matrix: ContextMatrix):
        """무작위 조합 생성 확인"""
        combo = context_matrix.sample_combination()
        assert isinstance(combo, ContextCombination)
        # 모든 필드가 채워져야 함
        d = combo.to_dict()
        filled_count = sum(1 for v in d.values() if v is not None)
        assert filled_count >= 8  # 대부분 필드가 채워져야 함

    def test_copy(self):
        """조합 복사 테스트"""
        original = ContextCombination(
            learner_age="20대",
            institution_type="대학교(학부)",
        )
        copied = original.copy()

        assert copied.learner_age == original.learner_age
        assert copied.institution_type == original.institution_type

        # 독립적인 복사본인지 확인
        copied.learner_age = "30대"
        assert original.learner_age == "20대"

    def test_get_set_field(self):
        """필드 조회/설정 테스트"""
        combo = ContextCombination()
        combo.set_field("learner_age", "10대")
        assert combo.get_field("learner_age") == "10대"


class TestRepresentativeScenarios:
    """대표 조합 생성 테스트"""

    def test_generate_representative(self, context_matrix: ContextMatrix):
        """대표 시나리오 생성"""
        scenarios = context_matrix.generate_representative_scenarios(n=50)
        assert len(scenarios) == 50
        assert all(isinstance(s, ContextCombination) for s in scenarios)

    def test_no_duplicates(self, context_matrix: ContextMatrix):
        """중복 없음 확인"""
        scenarios = context_matrix.generate_representative_scenarios(n=100)
        keys = [context_matrix._combination_key(s) for s in scenarios]
        assert len(keys) == len(set(keys))

    def test_includes_edge_cases(self, context_matrix: ContextMatrix):
        """Edge case 포함 확인"""
        scenarios = context_matrix.generate_representative_scenarios(
            n=50,
            include_edge_cases=True
        )
        # Edge case 중 하나라도 포함되어야 함
        has_vr = any(s.delivery_mode and "VR" in s.delivery_mode for s in scenarios)
        has_limited_tech = any(
            s.tech_environment and "제한적" in s.tech_environment
            for s in scenarios
        )
        assert has_vr or has_limited_tech


class TestAblationStudy:
    """Ablation Study 테스트"""

    def test_ablation_single_dimension(self, context_matrix: ContextMatrix):
        """단일 차원 Ablation"""
        base = ContextCombination(
            learner_age="20대",
            learner_education="대학",
            domain_expertise="중급",
            institution_type="대학교(학부)",
            education_domain="AI",
            delivery_mode="온라인 비실시간(LMS)",
            class_size="중규모(10–30명)",
            duration="중기 과정(2–4주)",
        )

        variants = context_matrix.generate_ablation_study(
            base,
            vary_dimension="학습자 특성"
        )

        # 학습자 특성 관련 필드들이 변경된 변형 생성
        assert len(variants) > 0

        # 모든 변형은 해당 차원의 필드만 변경됨
        for variant in variants:
            # 변경되지 않아야 할 필드 확인
            assert variant.institution_type == base.institution_type
            assert variant.delivery_mode == base.delivery_mode

    def test_ablation_preserves_other_fields(self, context_matrix: ContextMatrix):
        """Ablation 시 다른 필드 보존 확인 (Ceteris Paribus)"""
        base = ContextCombination(
            learner_age="20대",
            institution_type="대학교(학부)",
            delivery_mode="온라인 비실시간(LMS)",
        )

        # 연령만 변경
        age_options = context_matrix.get_options("학습자 특성", "연령")
        for age in age_options:
            if age != "20대":
                variant = base.copy()
                variant.set_field("learner_age", age)

                # 다른 필드는 그대로
                assert variant.institution_type == "대학교(학부)"
                assert variant.delivery_mode == "온라인 비실시간(LMS)"

    def test_full_ablation_study(self, context_matrix: ContextMatrix):
        """전체 Ablation Study"""
        bases = [
            ContextCombination(
                learner_age="20대",
                institution_type="대학교(학부)",
            ),
            ContextCombination(
                learner_age="30대",
                institution_type="기업",
            ),
        ]

        result = context_matrix.generate_full_ablation_study(bases)
        assert "base_0" in result
        assert "base_1" in result
        assert len(result["base_0"]) > 0


class TestAllCombinations:
    """전체 조합 테스트"""

    def test_count_combinations(self, context_matrix: ContextMatrix):
        """조합 수 계산"""
        count = context_matrix.count_all_combinations()
        # 4 * 5 * 3 * 4 * 6 * 10 * 7 * 3 * 3 * 3 * 3 = 수십만
        assert count > 100000

    def test_iterator_yields_combinations(self, context_matrix: ContextMatrix):
        """Iterator 동작 확인 (처음 몇 개만)"""
        count = 0
        for combo in context_matrix.all_combinations():
            assert isinstance(combo, ContextCombination)
            count += 1
            if count >= 10:
                break

        assert count == 10


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_csv_path(self):
        """빈 경로 처리"""
        matrix = ContextMatrix(csv_path=None)
        # 기본 경로가 존재하면 로드됨
        # 존재하지 않으면 빈 상태

    def test_default_path_load(self):
        """기본 경로 자동 로드"""
        matrix = ContextMatrix()
        # 기본 경로에 CSV가 있으면 로드됨
        if matrix.items:
            assert len(matrix.dimensions) == 5
