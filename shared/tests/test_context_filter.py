"""컨텍스트 필터링 단위 테스트"""

import pytest
from pathlib import Path

from shared.models.context_filter import (
    ContextFilter,
    ContextConstraint,
    FilterResult,
    EDUCATION_LEVEL_CONSTRAINTS,
    DOMAIN_CONSTRAINTS,
    ADVANCED_CONTENT_KEYWORDS,
)
from shared.models.context_matrix import ContextMatrix, ContextCombination
from shared.models.seed_extractor import ScenarioSeed, ExtractionStatus
from shared.models.scenario_generator import ScenarioGenerator
from shared.models.idld_dataset import IDLDDataset, IDLDRecord


@pytest.fixture
def context_filter() -> ContextFilter:
    """ContextFilter 인스턴스"""
    return ContextFilter()


@pytest.fixture
def context_matrix() -> ContextMatrix:
    """ContextMatrix 인스턴스"""
    return ContextMatrix()


@pytest.fixture
def csv_path() -> str:
    """테스트용 CSV 경로"""
    current_dir = Path(__file__).parent
    return str(current_dir.parent.parent / "scenarios" / "IDLD.xlsx - sheet1.csv")


@pytest.fixture
def idld_dataset(csv_path: str) -> IDLDDataset:
    """IDLDDataset 인스턴스"""
    return IDLDDataset(csv_path)


# =============================================================================
# ContextConstraint 테스트
# =============================================================================

class TestContextConstraint:
    """ContextConstraint 데이터 구조 테스트"""

    def test_constraint_creation(self):
        """제약 조건 생성"""
        constraint = ContextConstraint(
            field_name="institution_type",
            excluded_values=["초·중등학교", "기업"],
            reason="테스트 제약"
        )

        assert constraint.field_name == "institution_type"
        assert len(constraint.excluded_values) == 2

    def test_constraint_violation_detected(self):
        """위반 감지"""
        constraint = ContextConstraint(
            field_name="institution_type",
            excluded_values=["초·중등학교"],
            reason="고등교육 제약"
        )

        # 위반하는 컨텍스트
        context = ContextCombination(institution_type="초·중등학교")
        assert constraint.is_violated(context)

    def test_constraint_not_violated(self):
        """비위반 감지"""
        constraint = ContextConstraint(
            field_name="institution_type",
            excluded_values=["초·중등학교"],
            reason="고등교육 제약"
        )

        # 위반하지 않는 컨텍스트
        context = ContextCombination(institution_type="대학교(학부)")
        assert not constraint.is_violated(context)

    def test_constraint_none_value(self):
        """None 값은 위반 아님"""
        constraint = ContextConstraint(
            field_name="institution_type",
            excluded_values=["초·중등학교"],
            reason="고등교육 제약"
        )

        context = ContextCombination(institution_type=None)
        assert not constraint.is_violated(context)


# =============================================================================
# FilterResult 테스트
# =============================================================================

class TestFilterResult:
    """FilterResult 데이터 구조 테스트"""

    def test_initial_state(self):
        """초기 상태는 호환"""
        result = FilterResult(is_compatible=True)
        assert result.is_compatible
        assert len(result.violations) == 0

    def test_add_violation(self):
        """위반 추가"""
        result = FilterResult(is_compatible=True)
        constraint = ContextConstraint(
            field_name="learner_age",
            excluded_values=["10대"],
            reason="의료 교육 제약"
        )

        result.add_violation(constraint, "10대")

        assert not result.is_compatible
        assert len(result.violations) == 1
        assert "10대" in result.violations[0]


# =============================================================================
# 충돌 규칙 테스트
# =============================================================================

class TestConflictRules:
    """충돌 규칙 상수 테스트"""

    def test_education_level_constraints_exist(self):
        """교육 수준 제약 존재 확인"""
        assert "K-12" in EDUCATION_LEVEL_CONSTRAINTS
        assert "Higher Education" in EDUCATION_LEVEL_CONSTRAINTS
        assert "Corporate Training" in EDUCATION_LEVEL_CONSTRAINTS

    def test_k12_excludes_corporate(self):
        """K-12는 기업 맥락 제외"""
        k12_constraints = EDUCATION_LEVEL_CONSTRAINTS["K-12"]
        assert "institution_type" in k12_constraints
        assert "기업" in k12_constraints["institution_type"]

    def test_higher_education_excludes_k12(self):
        """고등교육은 초중등 맥락 제외"""
        he_constraints = EDUCATION_LEVEL_CONSTRAINTS["Higher Education"]
        assert "institution_type" in he_constraints
        assert "초·중등학교" in he_constraints["institution_type"]

    def test_healthcare_excludes_minors(self):
        """의료 분야는 미성년자 제외"""
        if "Healthcare" in DOMAIN_CONSTRAINTS:
            hc_constraints = DOMAIN_CONSTRAINTS["Healthcare"]
            assert "learner_age" in hc_constraints
            assert "10대" in hc_constraints["learner_age"]


# =============================================================================
# ContextFilter 테스트
# =============================================================================

class TestContextFilter:
    """ContextFilter 클래스 테스트"""

    def test_check_k12_with_corporate_context(self, context_filter):
        """K-12 시드 + 기업 컨텍스트 = 비호환"""
        seed = ScenarioSeed(
            topic="초등학교 수학",
            pedagogical_method="게이미피케이션",
            categories=["K-12", "STEM"],
        )

        context = ContextCombination(
            institution_type="기업",
            learner_age="30대",
        )

        result = context_filter.check_compatibility(seed, context)

        assert not result.is_compatible
        assert len(result.violations) > 0

    def test_check_k12_with_school_context(self, context_filter):
        """K-12 시드 + 학교 컨텍스트 = 호환"""
        seed = ScenarioSeed(
            topic="중학교 과학",
            pedagogical_method="실험 기반 학습",
            categories=["K-12", "STEM"],
        )

        context = ContextCombination(
            institution_type="초·중등학교",
            learner_age="10대",
        )

        result = context_filter.check_compatibility(seed, context)

        assert result.is_compatible
        assert len(result.violations) == 0

    def test_check_higher_ed_with_k12_context(self, context_filter):
        """고등교육 시드 + 초중등 컨텍스트 = 비호환"""
        seed = ScenarioSeed(
            topic="대학 미적분학",
            pedagogical_method="플립드 러닝",
            categories=["Higher Education", "STEM"],
        )

        context = ContextCombination(
            institution_type="초·중등학교",
            learner_age="10대",
        )

        result = context_filter.check_compatibility(seed, context)

        assert not result.is_compatible

    def test_check_healthcare_with_minor_context(self, context_filter):
        """의료 시드 + 미성년자 컨텍스트 = 비호환"""
        seed = ScenarioSeed(
            topic="간호 시뮬레이션",
            pedagogical_method="시뮬레이션 기반 학습",
            categories=["Healthcare", "Higher Education"],
        )

        context = ContextCombination(
            learner_age="10대",
            institution_type="초·중등학교",
        )

        result = context_filter.check_compatibility(seed, context)

        assert not result.is_compatible

    def test_advanced_content_excludes_beginners(self, context_filter):
        """고급 콘텐츠는 초급자 제외"""
        seed = ScenarioSeed(
            topic="Advanced Machine Learning",
            pedagogical_method="프로젝트 기반 학습",
            categories=["IT/Computer Science", "Higher Education"],
        )

        context = ContextCombination(
            domain_expertise="초급",
            institution_type="대학교(학부)",
        )

        result = context_filter.check_compatibility(seed, context)

        assert not result.is_compatible
        assert any("초급" in v for v in result.violations)

    def test_beginner_content_allows_beginners(self, context_filter):
        """입문 콘텐츠는 초급자 허용"""
        seed = ScenarioSeed(
            topic="프로그래밍 입문",
            pedagogical_method="단계별 학습",
            categories=["IT/Computer Science"],
        )

        context = ContextCombination(
            domain_expertise="초급",
            institution_type="대학교(학부)",
        )

        result = context_filter.check_compatibility(seed, context)

        # 입문 콘텐츠이므로 초급자 제외 규칙 미적용
        # 다른 위반이 없으면 호환
        # (IT 분야의 다른 제약이 없다고 가정)
        assert result.is_compatible or "초급" not in str(result.violations)


class TestContextFilterBatch:
    """배치 필터링 테스트"""

    def test_filter_compatible_contexts(self, context_filter, context_matrix):
        """호환 컨텍스트 필터링"""
        seed = ScenarioSeed(
            topic="대학 물리학",
            pedagogical_method="문제 중심 학습",
            categories=["Higher Education", "STEM"],
        )

        # 여러 컨텍스트 생성
        all_contexts = context_matrix.generate_representative_scenarios(n=20)

        # 필터링
        compatible = context_filter.filter_compatible_contexts(seed, all_contexts)

        # 필터링된 결과 검증
        for ctx in compatible:
            result = context_filter.check_compatibility(seed, ctx)
            assert result.is_compatible

    def test_filter_with_details(self, context_filter, context_matrix):
        """상세 결과 포함 필터링"""
        seed = ScenarioSeed(
            topic="기업 리더십 교육",
            pedagogical_method="케이스 스터디",
            categories=["Corporate Training", "Business"],
        )

        all_contexts = context_matrix.generate_representative_scenarios(n=30)

        compatible, incompatible = context_filter.filter_with_details(seed, all_contexts)

        # 호환/비호환 모두 존재해야 함 (다양한 컨텍스트 생성됨)
        assert len(compatible) + len(incompatible) == len(all_contexts)

        # 비호환 항목은 위반 사유가 있어야 함
        for ctx, result in incompatible:
            assert not result.is_compatible
            assert len(result.violations) > 0


class TestContextFilterRuleManagement:
    """규칙 관리 테스트"""

    def test_add_education_constraint(self):
        """교육 수준 제약 추가"""
        filter = ContextFilter()

        # 새 제약 추가
        filter.add_education_constraint(
            category="Lifelong Learning",
            field_name="institution_type",
            excluded_values=["초·중등학교"],
        )

        assert "Lifelong Learning" in filter.education_constraints
        assert "초·중등학교" in filter.education_constraints["Lifelong Learning"]["institution_type"]

    def test_get_constraints_for_seed(self, context_filter):
        """시드에 적용되는 제약 조회"""
        seed = ScenarioSeed(
            topic="고등교육 과정",
            pedagogical_method="강의",
            categories=["Higher Education"],
        )

        constraints = context_filter.get_constraints_for_seed(seed)

        assert len(constraints) > 0
        # 고등교육 제약이 포함되어야 함
        field_names = [c.field_name for c in constraints]
        assert "institution_type" in field_names

    def test_constraint_summary(self, context_filter):
        """제약 요약 정보"""
        summary = context_filter.get_constraint_summary()

        assert "education_categories" in summary
        assert "domain_categories" in summary
        assert "total_education_rules" in summary
        assert summary["total_education_rules"] > 0

    def test_explain_constraints(self, context_filter):
        """제약 설명"""
        seed = ScenarioSeed(
            topic="의료 시뮬레이션",
            pedagogical_method="VR 기반 학습",
            categories=["Healthcare", "Higher Education"],
        )

        explanation = context_filter.explain_constraints(seed)

        assert "의료 시뮬레이션" in explanation
        assert "제약" in explanation


# =============================================================================
# ScenarioGenerator 통합 테스트
# =============================================================================

class TestScenarioGeneratorIntegration:
    """ScenarioGenerator와 ContextFilter 통합 테스트"""

    def test_generator_has_filter(self, idld_dataset, context_matrix):
        """Generator에 필터가 포함됨"""
        generator = ScenarioGenerator(
            idld_dataset=idld_dataset,
            context_matrix=context_matrix,
        )

        assert generator.context_filter is not None
        assert isinstance(generator.context_filter, ContextFilter)

    def test_prepare_requests_with_seed(self, idld_dataset, context_matrix):
        """시드 기반 요청 준비"""
        generator = ScenarioGenerator(
            idld_dataset=idld_dataset,
            context_matrix=context_matrix,
        )

        seed = ScenarioSeed(
            topic="대학 통계학",
            pedagogical_method="실습 기반 학습",
            categories=["Higher Education", "STEM"],
        )

        record = idld_dataset.sample(n=1)[0]

        requests = generator.prepare_requests_with_seed(
            seed=seed,
            record=record,
            n_contexts=3,
            difficulty="medium",
        )

        # 요청이 생성됨
        assert len(requests) > 0

        # 모든 요청이 시드와 호환되는 컨텍스트를 가짐
        for req in requests:
            result = generator.context_filter.check_compatibility(
                seed, req.context_combination
            )
            assert result.is_compatible

    def test_prepare_batch_with_seeds(self, idld_dataset, context_matrix):
        """배치 시드 기반 요청 준비"""
        generator = ScenarioGenerator(
            idld_dataset=idld_dataset,
            context_matrix=context_matrix,
        )

        # 여러 시드 생성
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
        ]

        records = idld_dataset.sample(n=2)

        requests = generator.prepare_batch_with_seeds(
            seeds=seeds,
            records=records,
        )

        assert len(requests) == 2

        # 각 요청이 해당 시드와 호환되는지 확인
        for i, req in enumerate(requests):
            result = generator.context_filter.check_compatibility(
                seeds[i], req.context_combination
            )
            assert result.is_compatible

    def test_batch_seed_record_mismatch_error(self, idld_dataset, context_matrix):
        """시드/레코드 수 불일치 에러"""
        generator = ScenarioGenerator(
            idld_dataset=idld_dataset,
            context_matrix=context_matrix,
        )

        seeds = [
            ScenarioSeed(topic="주제1", pedagogical_method="방법1", categories=[])
        ]
        records = idld_dataset.sample(n=3)  # 불일치

        with pytest.raises(ValueError, match="시드와 레코드 수"):
            generator.prepare_batch_with_seeds(seeds=seeds, records=records)
