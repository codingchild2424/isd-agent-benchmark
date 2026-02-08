"""프롬프트 빌더 단위 테스트"""

import pytest
from pathlib import Path

from shared.models.prompt_builder import (
    PromptBuilder,
    PromptBuildResult,
    Language,
    TEMPLATES,
    DIFFICULTY_DESCRIPTIONS,
)
from shared.models.context_matrix import ContextMatrix, ContextCombination
from shared.models.seed_extractor import ScenarioSeed
from shared.models.idld_dataset import IDLDDataset, IDLDRecord


@pytest.fixture
def prompt_builder() -> PromptBuilder:
    """PromptBuilder 인스턴스"""
    return PromptBuilder()


@pytest.fixture
def sample_seed() -> ScenarioSeed:
    """샘플 시드"""
    return ScenarioSeed(
        topic="VR 기반 생화학 학습",
        pedagogical_method="시뮬레이션 기반 학습",
        categories=["Higher Education", "STEM"],
    )


@pytest.fixture
def sample_context() -> ContextCombination:
    """샘플 컨텍스트"""
    return ContextCombination(
        learner_age="20대",
        learner_education="대학생",
        domain_expertise="중급",
        learner_role="학생/취준생",
        institution_type="대학교(학부)",
        education_domain="과학",
        delivery_mode="시뮬레이션/VR 기반",
        class_size="중규모(10–30명)",
        evaluation_focus="형성평가 중심",
        tech_environment="LMS/디지털 도구 활용 가능",
        duration="중기 과정(2–4주)",
    )


@pytest.fixture
def sample_record() -> IDLDRecord:
    """샘플 IDLD 레코드"""
    return IDLDRecord(
        no=1,
        year=2023,
        title="VR in Biochemistry Education",
        abstract="""This study investigates the effectiveness of virtual reality (VR)
        in teaching biochemistry concepts to undergraduate students. Using immersive
        3D models of molecular structures, students demonstrated improved spatial
        understanding and retention of metabolic pathways. The results suggest that
        VR-based simulations can significantly enhance learning outcomes in
        challenging scientific domains.""",
        keywords={"virtual reality", "biochemistry", "higher education", "stem"},
    )


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
# 템플릿 테스트
# =============================================================================

class TestTemplates:
    """템플릿 상수 테스트"""

    def test_templates_exist(self):
        """템플릿 존재 확인"""
        assert "ko" in TEMPLATES
        assert "en" in TEMPLATES

    def test_korean_template_has_slots(self):
        """한국어 템플릿 슬롯 확인"""
        template = TEMPLATES["ko"]
        assert "{topic}" in template
        assert "{pedagogical_method}" in template
        assert "{learner_age}" in template
        assert "{institution_type}" in template
        assert "{difficulty}" in template

    def test_english_template_has_slots(self):
        """영어 템플릿 슬롯 확인"""
        template = TEMPLATES["en"]
        assert "{topic}" in template
        assert "{pedagogical_method}" in template
        assert "{learner_age}" in template

    def test_difficulty_descriptions(self):
        """난이도 설명 확인"""
        assert "ko" in DIFFICULTY_DESCRIPTIONS
        assert "en" in DIFFICULTY_DESCRIPTIONS
        assert "easy" in DIFFICULTY_DESCRIPTIONS["ko"]
        assert "medium" in DIFFICULTY_DESCRIPTIONS["ko"]
        assert "hard" in DIFFICULTY_DESCRIPTIONS["ko"]


# =============================================================================
# PromptBuildResult 테스트
# =============================================================================

class TestPromptBuildResult:
    """PromptBuildResult 데이터 구조 테스트"""

    def test_valid_result(self):
        """유효한 결과"""
        result = PromptBuildResult(
            prompt="Test prompt",
            language="ko",
            variables={"topic": "test"},
            token_estimate=100,
            warnings=[],
        )

        assert result.is_valid
        assert result.token_estimate == 100

    def test_invalid_result_with_warnings(self):
        """경고가 있는 결과"""
        result = PromptBuildResult(
            prompt="Test prompt",
            language="ko",
            variables={},
            token_estimate=5000,
            warnings=["토큰 초과"],
        )

        assert not result.is_valid

    def test_invalid_result_empty_prompt(self):
        """빈 프롬프트"""
        result = PromptBuildResult(
            prompt="",
            language="ko",
            variables={},
            token_estimate=0,
            warnings=[],
        )

        assert not result.is_valid


# =============================================================================
# PromptBuilder 기본 기능 테스트
# =============================================================================

class TestPromptBuilderBasic:
    """PromptBuilder 기본 기능 테스트"""

    def test_build_korean_prompt(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """한국어 프롬프트 생성"""
        result = prompt_builder.build(
            seed=sample_seed,
            context=sample_context,
            record=sample_record,
            difficulty="medium",
            language="ko",
        )

        assert result.is_valid
        assert result.language == "ko"
        assert sample_seed.topic in result.prompt
        assert sample_seed.pedagogical_method in result.prompt
        assert sample_context.institution_type in result.prompt

    def test_build_english_prompt(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """영어 프롬프트 생성"""
        result = prompt_builder.build(
            seed=sample_seed,
            context=sample_context,
            record=sample_record,
            difficulty="easy",
            language="en",
        )

        assert result.is_valid
        assert result.language == "en"
        assert sample_seed.topic in result.prompt

    def test_all_variables_injected(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """모든 변수 주입 확인"""
        result = prompt_builder.build(
            seed=sample_seed,
            context=sample_context,
            record=sample_record,
            difficulty="hard",
            language="ko",
        )

        # 변수 딕셔너리 확인
        assert "topic" in result.variables
        assert "pedagogical_method" in result.variables
        assert "learner_age" in result.variables
        assert "institution_type" in result.variables

        # 프롬프트에 플레이스홀더가 남아있지 않아야 함
        assert "{topic}" not in result.prompt
        assert "{learner_age}" not in result.prompt

    def test_context_values_preserved(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """컨텍스트 값이 그대로 유지되는지 확인 (데이터 호환성)"""
        result = prompt_builder.build(
            seed=sample_seed,
            context=sample_context,
            record=sample_record,
            language="ko",
        )

        # 원래 컨텍스트 값이 프롬프트에 그대로 포함되어야 함
        assert "대학교(학부)" in result.prompt
        assert "20대" in result.prompt
        assert "시뮬레이션/VR 기반" in result.prompt


class TestPromptBuilderValidation:
    """PromptBuilder 검증 기능 테스트"""

    def test_long_abstract_truncation(self, prompt_builder, sample_seed, sample_context):
        """긴 초록 잘라내기"""
        long_abstract = "A" * 3000  # 매우 긴 초록
        record = IDLDRecord(
            no=1,
            year=2023,
            title="Test",
            abstract=long_abstract,
        )

        result = prompt_builder.build(
            seed=sample_seed,
            context=sample_context,
            record=record,
            language="ko",
        )

        # 잘라내기 경고가 있어야 함
        assert any("잘랐습니다" in w for w in result.warnings)

    def test_validate_prompt_length_ok(self, prompt_builder):
        """프롬프트 길이 검증 - 정상"""
        short_prompt = "짧은 프롬프트" * 100

        is_valid, tokens, msg = prompt_builder.validate_prompt_length(
            short_prompt, "ko"
        )

        assert is_valid
        assert msg == "OK"

    def test_validate_prompt_length_exceeded(self):
        """프롬프트 길이 검증 - 초과"""
        builder = PromptBuilder(max_tokens=100)
        long_prompt = "A" * 1000

        is_valid, tokens, msg = builder.validate_prompt_length(
            long_prompt, "ko"
        )

        assert not is_valid
        assert "초과" in msg

    def test_token_estimation_korean(self, prompt_builder):
        """한국어 토큰 추정"""
        text = "가나다라마바사"  # 7자
        tokens = prompt_builder._estimate_tokens(text, "ko")

        # 한글 ~2자/토큰
        assert 3 <= tokens <= 4

    def test_token_estimation_english(self, prompt_builder):
        """영어 토큰 추정"""
        text = "Hello World Test"  # 16자
        tokens = prompt_builder._estimate_tokens(text, "en")

        # 영어 ~4자/토큰
        assert 3 <= tokens <= 5


class TestPromptBuilderBatch:
    """배치 처리 테스트"""

    def test_build_batch(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """배치 프롬프트 생성"""
        seeds = [sample_seed, sample_seed]
        contexts = [sample_context, sample_context]
        records = [sample_record, sample_record]

        results = prompt_builder.build_batch(
            seeds=seeds,
            contexts=contexts,
            records=records,
            difficulties=["easy", "hard"],
            language="ko",
        )

        assert len(results) == 2
        assert results[0].is_valid
        assert results[1].is_valid

    def test_build_batch_default_difficulty(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """배치 생성 기본 난이도"""
        results = prompt_builder.build_batch(
            seeds=[sample_seed],
            contexts=[sample_context],
            records=[sample_record],
            language="ko",
        )

        assert len(results) == 1
        # 기본 난이도 "medium"이 적용됨
        assert "보통" in results[0].prompt

    def test_build_batch_length_mismatch_error(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """배치 생성 길이 불일치 에러"""
        with pytest.raises(ValueError, match="길이가 일치"):
            prompt_builder.build_batch(
                seeds=[sample_seed],
                contexts=[sample_context, sample_context],
                records=[sample_record],
            )


class TestPromptBuilderTemplateManagement:
    """템플릿 관리 테스트"""

    def test_set_custom_template(self, prompt_builder):
        """커스텀 템플릿 설정"""
        custom = "Custom template: {topic}"
        prompt_builder.set_template("custom", custom)

        assert prompt_builder.get_template("custom") == custom

    def test_list_languages(self, prompt_builder):
        """지원 언어 목록"""
        languages = prompt_builder.list_languages()

        assert "ko" in languages
        assert "en" in languages

    def test_explain_template(self, prompt_builder):
        """템플릿 설명"""
        explanation = prompt_builder.explain_template("ko")

        assert "변수 슬롯" in explanation
        assert "{topic}" in explanation


class TestPromptBuilderPreview:
    """미리보기 기능 테스트"""

    def test_preview_variables(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """변수 미리보기"""
        variables = prompt_builder.preview_variables(
            seed=sample_seed,
            context=sample_context,
            record=sample_record,
            difficulty="medium",
            language="ko",
        )

        assert variables["topic"] == sample_seed.topic
        assert variables["learner_age"] == sample_context.learner_age
        assert variables["institution_type"] == sample_context.institution_type


class TestPromptBuilderEdgeCases:
    """엣지 케이스 테스트"""

    def test_none_context_values(self, prompt_builder, sample_seed, sample_record):
        """컨텍스트 None 값 처리"""
        empty_context = ContextCombination()  # 모든 값 None

        result = prompt_builder.build(
            seed=sample_seed,
            context=empty_context,
            record=sample_record,
            language="ko",
        )

        # None 값은 "미지정"으로 대체
        assert "미지정" in result.prompt
        assert result.is_valid

    def test_empty_keywords(self, prompt_builder, sample_seed, sample_context):
        """빈 키워드 처리"""
        record = IDLDRecord(
            no=1,
            year=2023,
            title="Test",
            abstract="Test abstract",
            keywords=set(),  # 빈 키워드
        )

        result = prompt_builder.build(
            seed=sample_seed,
            context=sample_context,
            record=record,
            language="ko",
        )

        assert result.is_valid

    def test_difficulty_descriptions_in_prompt(
        self, prompt_builder, sample_seed, sample_context, sample_record
    ):
        """난이도 설명이 프롬프트에 포함"""
        for difficulty in ["easy", "medium", "hard"]:
            result = prompt_builder.build(
                seed=sample_seed,
                context=sample_context,
                record=sample_record,
                difficulty=difficulty,
                language="ko",
            )

            # 난이도 설명이 프롬프트에 포함되어야 함
            desc = DIFFICULTY_DESCRIPTIONS["ko"][difficulty]
            assert desc in result.prompt


class TestPromptBuilderIntegration:
    """통합 테스트"""

    def test_with_real_dataset(
        self, prompt_builder, idld_dataset
    ):
        """실제 데이터셋과 통합"""
        # 실제 레코드 샘플
        records = idld_dataset.sample(n=2)

        for record in records:
            seed = ScenarioSeed(
                topic="테스트 주제",
                pedagogical_method="테스트 방법",
                categories=["STEM"],
            )

            context = ContextCombination(
                learner_age="20대",
                institution_type="대학교(학부)",
            )

            result = prompt_builder.build(
                seed=seed,
                context=context,
                record=record,
                language="ko",
            )

            # 프롬프트가 생성되었으면 성공 (잘라내기 경고는 허용)
            assert len(result.prompt) > 0
            assert result.token_estimate > 0
            # 토큰 제한 초과 경고는 없어야 함
            assert not any("초과" in w for w in result.warnings)
