"""SmartSelector 단위 테스트"""

import pytest
from pathlib import Path

from shared.models.smart_selector import (
    SmartSelector,
    VariantResult,
    INSTITUTION_LEARNER_PAIRS,
    DELIVERY_COMBINATIONS,
    CHALLENGING_COMBINATIONS,
)
from shared.models.context_matrix import ContextMatrix, ContextCombination
from shared.models.context_filter import ContextFilter
from shared.models.seed_extractor import ScenarioSeed


@pytest.fixture
def smart_selector() -> SmartSelector:
    """SmartSelector 인스턴스"""
    return SmartSelector()


@pytest.fixture
def context_matrix() -> ContextMatrix:
    """ContextMatrix 인스턴스"""
    return ContextMatrix()


@pytest.fixture
def context_filter() -> ContextFilter:
    """ContextFilter 인스턴스"""
    return ContextFilter()


# =============================================================================
# 상수 테스트
# =============================================================================

class TestConstants:
    """상수 정의 테스트"""

    def test_institution_learner_pairs_exist(self):
        """기관-학습자 매칭 쌍 존재"""
        assert len(INSTITUTION_LEARNER_PAIRS) > 0
        # 각 쌍은 3개 요소 (기관, 연령, 역할)
        for pair in INSTITUTION_LEARNER_PAIRS:
            assert len(pair) == 3

    def test_delivery_combinations_exist(self):
        """전달방식 조합 존재"""
        assert len(DELIVERY_COMBINATIONS) > 0
        # 각 조합은 3개 요소 (전달방식, 규모, 기간)
        for combo in DELIVERY_COMBINATIONS:
            assert len(combo) == 3

    def test_challenging_combinations_exist(self):
        """도전적 조합 존재"""
        assert len(CHALLENGING_COMBINATIONS) > 0
        for combo in CHALLENGING_COMBINATIONS:
            assert isinstance(combo, dict)
            assert len(combo) > 0


# =============================================================================
# VariantResult 테스트
# =============================================================================

class TestVariantResult:
    """VariantResult 데이터 구조 테스트"""

    def test_result_creation(self):
        """결과 객체 생성"""
        seed = ScenarioSeed(
            topic="테스트 주제",
            pedagogical_method="테스트 방법",
            categories=["Higher Education"],
        )
        variants = [ContextCombination()]

        result = VariantResult(
            variants=variants,
            seed=seed,
            total_generated=10,
            filtered_count=3,
            duplicate_count=2,
        )

        assert result.success_count == 1
        assert result.total_generated == 10

    def test_result_summary(self):
        """결과 요약"""
        seed = ScenarioSeed(
            topic="머신러닝 기초",
            pedagogical_method="프로젝트 학습",
            categories=[],
        )

        result = VariantResult(
            variants=[ContextCombination(), ContextCombination()],
            seed=seed,
            total_generated=20,
            filtered_count=5,
            duplicate_count=3,
        )

        summary = result.summary()

        assert summary["seed_topic"] == "머신러닝 기초"
        assert summary["total_generated"] == 20
        assert summary["filtered_by_rules"] == 5
        assert summary["duplicates_removed"] == 3
        assert summary["final_variants"] == 2


# =============================================================================
# SmartSelector 기본 테스트
# =============================================================================

class TestSmartSelectorBasic:
    """SmartSelector 기본 기능 테스트"""

    def test_selector_creation(self):
        """선택기 생성"""
        selector = SmartSelector()

        assert selector.context_matrix is not None
        assert selector.context_filter is not None

    def test_selector_with_custom_components(self, context_matrix, context_filter):
        """커스텀 컴포넌트로 생성"""
        selector = SmartSelector(
            context_matrix=context_matrix,
            context_filter=context_filter,
        )

        assert selector.context_matrix is context_matrix
        assert selector.context_filter is context_filter

    def test_strategy_summary(self, smart_selector):
        """전략 요약"""
        summary = smart_selector.get_strategy_summary()

        assert "institution_learner_pairs" in summary
        assert "delivery_combinations" in summary
        assert "challenging_combinations" in summary
        assert summary["institution_learner_pairs"] > 0


# =============================================================================
# 변형 생성 테스트
# =============================================================================

class TestGenerateVariants:
    """변형 생성 테스트"""

    def test_generate_variants_basic(self, smart_selector):
        """기본 변형 생성"""
        seed = ScenarioSeed(
            topic="데이터 분석 기초",
            pedagogical_method="실습 기반 학습",
            categories=["Higher Education", "IT/Computer Science"],
        )

        result = smart_selector.generate_variants(seed, n=5)

        assert isinstance(result, VariantResult)
        assert result.success_count > 0
        assert result.success_count <= 5

    def test_generate_variants_respects_filter(self, smart_selector):
        """변형이 필터 규칙 준수"""
        seed = ScenarioSeed(
            topic="대학원 연구방법론",
            pedagogical_method="세미나 기반",
            categories=["Graduate", "Education"],
        )

        result = smart_selector.generate_variants(seed, n=10)

        # 모든 변형이 필터를 통과해야 함
        for variant in result.variants:
            filter_result = smart_selector.context_filter.check_compatibility(
                seed, variant
            )
            assert filter_result.is_compatible

    def test_generate_variants_no_duplicates(self, smart_selector):
        """변형에 중복 없음"""
        seed = ScenarioSeed(
            topic="웹 개발",
            pedagogical_method="프로젝트 기반",
            categories=["IT/Computer Science"],
        )

        result = smart_selector.generate_variants(seed, n=15)

        # 모든 변형이 고유해야 함
        keys = set()
        for variant in result.variants:
            key = "|".join(str(v) for v in variant.to_dict().values())
            assert key not in keys, "중복 변형 발견"
            keys.add(key)

    def test_generate_variants_with_challenging(self, smart_selector):
        """도전적 조합 포함"""
        seed = ScenarioSeed(
            topic="기초 프로그래밍",
            pedagogical_method="단계별 학습",
            categories=["IT/Computer Science"],
        )

        result = smart_selector.generate_variants(
            seed, n=20, include_challenging=True
        )

        # 다양한 조합이 포함되어야 함
        assert result.total_generated > result.success_count

    def test_generate_variants_without_challenging(self, smart_selector):
        """도전적 조합 제외"""
        seed = ScenarioSeed(
            topic="비즈니스 영어",
            pedagogical_method="롤플레이",
            categories=["Language", "Corporate Training"],
        )

        result = smart_selector.generate_variants(
            seed, n=10, include_challenging=False
        )

        assert result.success_count > 0


# =============================================================================
# 배치 생성 테스트
# =============================================================================

class TestGenerateBatch:
    """배치 변형 생성 테스트"""

    def test_generate_batch_basic(self, smart_selector):
        """기본 배치 생성"""
        seeds = [
            ScenarioSeed(
                topic="주제1",
                pedagogical_method="방법1",
                categories=["Higher Education"],
            ),
            ScenarioSeed(
                topic="주제2",
                pedagogical_method="방법2",
                categories=["K-12"],
            ),
            ScenarioSeed(
                topic="주제3",
                pedagogical_method="방법3",
                categories=["Corporate Training"],
            ),
        ]

        results = smart_selector.generate_batch(
            seeds, variants_per_seed=3
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, VariantResult)
            assert result.success_count > 0

    def test_generate_batch_different_categories(self, smart_selector):
        """다른 카테고리 시드 배치 생성"""
        seeds = [
            ScenarioSeed(
                topic="초등 수학",
                pedagogical_method="게임 기반",
                categories=["K-12", "STEM"],
            ),
            ScenarioSeed(
                topic="간호 실습",
                pedagogical_method="시뮬레이션",
                categories=["Healthcare", "Higher Education"],
            ),
        ]

        results = smart_selector.generate_batch(seeds, variants_per_seed=5)

        # 각 시드에 맞는 변형이 생성됨
        for i, result in enumerate(results):
            seed = seeds[i]
            for variant in result.variants:
                filter_result = smart_selector.context_filter.check_compatibility(
                    seed, variant
                )
                assert filter_result.is_compatible


# =============================================================================
# 지정 컨텍스트 변형 테스트
# =============================================================================

class TestGenerateVariantSet:
    """지정 컨텍스트 변형 생성 테스트"""

    def test_generate_variant_set_basic(self, smart_selector):
        """지정 컨텍스트로 변형 생성"""
        seed = ScenarioSeed(
            topic="머신러닝 기초 교육",
            pedagogical_method="실습 기반",
            categories=["IT/Computer Science"],
        )

        # V1~V4 예시와 유사한 타겟 컨텍스트
        target_contexts = [
            {
                "learner_age": "20대",
                "institution_type": "대학교(학부)",
                "delivery_mode": "블렌디드(혼합형)",
            },
            {
                "learner_age": "30대",
                "institution_type": "기업",
                "delivery_mode": "온라인 비실시간(LMS)",
            },
            {
                "learner_age": "20대",
                "institution_type": "대학원",
                "delivery_mode": "자기주도 학습",
            },
        ]

        variants = smart_selector.generate_variant_set(seed, target_contexts)

        assert len(variants) > 0
        # 각 변형이 지정된 필드를 가지고 있어야 함
        for variant in variants:
            d = variant.to_dict()
            # 최소한 일부 지정 필드가 적용됨
            assert d["learner_age"] in ["20대", "30대"]

    def test_generate_variant_set_filters_incompatible(self, smart_selector):
        """비호환 지정 컨텍스트는 필터링됨"""
        seed = ScenarioSeed(
            topic="대학원 고급 연구",
            pedagogical_method="세미나",
            categories=["Graduate"],
        )

        # 일부는 호환, 일부는 비호환
        target_contexts = [
            {
                "learner_age": "20대",
                "institution_type": "대학원",
            },
            {
                "learner_age": "10대",  # Graduate와 비호환
                "institution_type": "초·중등학교",  # Graduate와 비호환
            },
        ]

        variants = smart_selector.generate_variant_set(seed, target_contexts)

        # 비호환 조합은 제외됨
        for variant in variants:
            result = smart_selector.context_filter.check_compatibility(seed, variant)
            assert result.is_compatible


# =============================================================================
# 설명 기능 테스트
# =============================================================================

class TestExplainVariants:
    """변형 설명 테스트"""

    def test_explain_variants(self, smart_selector):
        """변형 결과 설명"""
        seed = ScenarioSeed(
            topic="파이썬 기초",
            pedagogical_method="코딩 실습",
            categories=["IT/Computer Science"],
        )

        result = smart_selector.generate_variants(seed, n=3)
        explanation = smart_selector.explain_variants(result)

        assert "파이썬 기초" in explanation
        assert "생성된 변형" in explanation


# =============================================================================
# 실제 시나리오 테스트
# =============================================================================

class TestRealScenarios:
    """실제 시나리오 기반 테스트"""

    def test_ml_education_variants(self, smart_selector):
        """머신러닝 교육 변형 생성 예시"""
        seed = ScenarioSeed(
            topic="머신러닝 기초 교육",
            pedagogical_method="프로젝트 기반 학습",
            categories=["IT/Computer Science", "Higher Education"],
        )

        result = smart_selector.generate_variants(seed, n=10)

        # 다양한 기관 유형이 포함되어야 함
        institutions = set()
        for variant in result.variants:
            if variant.institution_type:
                institutions.add(variant.institution_type)

        # 최소 2개 이상의 다른 기관 유형
        assert len(institutions) >= 2

        # 다양한 학습자 연령이 포함되어야 함
        ages = set()
        for variant in result.variants:
            if variant.learner_age:
                ages.add(variant.learner_age)

        assert len(ages) >= 2

    def test_healthcare_education_variants(self, smart_selector):
        """의료 교육 시나리오 - 필터링 확인"""
        seed = ScenarioSeed(
            topic="간호 시뮬레이션 실습",
            pedagogical_method="VR 기반 학습",
            categories=["Healthcare", "Higher Education"],
        )

        result = smart_selector.generate_variants(seed, n=10)

        # 모든 변형이 10대 학습자를 포함하지 않아야 함
        for variant in result.variants:
            assert variant.learner_age != "10대"

        # 모든 변형이 초·중등학교를 포함하지 않아야 함
        for variant in result.variants:
            assert variant.institution_type != "초·중등학교"

    def test_k12_education_variants(self, smart_selector):
        """K-12 교육 시나리오 - 필터링 확인"""
        seed = ScenarioSeed(
            topic="초등학교 수학 게임",
            pedagogical_method="게이미피케이션",
            categories=["K-12", "STEM"],
        )

        result = smart_selector.generate_variants(seed, n=10)

        # 모든 변형이 기업을 포함하지 않아야 함
        for variant in result.variants:
            assert variant.institution_type != "기업"

        # 모든 변형이 30대 이상을 포함하지 않아야 함
        for variant in result.variants:
            assert variant.learner_age not in ["30대", "40대 이상"]
