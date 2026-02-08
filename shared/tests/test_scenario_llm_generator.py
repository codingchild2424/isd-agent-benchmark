"""ScenarioLLMGenerator 단위 테스트"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from shared.models.scenario_llm_generator import (
    ScenarioLLMGenerator,
    GenerationResult,
    BatchGenerationResult,
    GenerationStatus,
    LLMScenarioOutput,
    ContextSchema,
    ConstraintsSchema,
)
from shared.models.prompt_builder import PromptBuildResult
from shared.models.idld_dataset import IDLDRecord, ScenarioSchema


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_prompt_result() -> PromptBuildResult:
    """샘플 프롬프트 결과"""
    return PromptBuildResult(
        prompt="테스트 프롬프트 내용",
        language="ko",
        variables={"topic": "VR 학습"},
        token_estimate=500,
        warnings=[],
    )


@pytest.fixture
def sample_record() -> IDLDRecord:
    """샘플 IDLD 레코드"""
    return IDLDRecord(
        no=1,
        year=2024,
        title="Test Paper",
        abstract="This is a test abstract about VR in education.",
        keywords={"VR", "education"},
    )


@pytest.fixture
def valid_llm_response() -> str:
    """유효한 LLM JSON 응답"""
    return '''```json
{
  "title": "VR 기반 과학 실험 교육",
  "context": {
    "target_audience": "고등학생 1-2학년",
    "prior_knowledge": "중학교 과학 이수",
    "duration": "2시간",
    "learning_environment": "VR 체험실",
    "class_size": 20,
    "additional_context": "과학 동아리 활동"
  },
  "learning_goals": [
    "분자 구조를 3D로 관찰할 수 있다",
    "VR 조작법을 익힌다"
  ],
  "constraints": {
    "budget": "medium",
    "resources": ["VR 헤드셋", "시뮬레이션 앱"],
    "accessibility": null,
    "language": "ko"
  },
  "difficulty": "easy",
  "domain": "STEM/과학"
}
```'''


@pytest.fixture
def invalid_json_response() -> str:
    """잘못된 JSON 응답"""
    return "이것은 유효한 JSON이 아닙니다."


@pytest.fixture
def incomplete_json_response() -> str:
    """불완전한 JSON 응답 (필수 필드 누락)"""
    return '''```json
{
  "title": "테스트",
  "context": {
    "target_audience": "학생"
  }
}
```'''


# =============================================================================
# JSON 추출 테스트
# =============================================================================

class TestJSONExtraction:
    """JSON 추출 로직 테스트"""

    def test_extract_json_from_markdown_block(self):
        """마크다운 코드 블록에서 JSON 추출"""
        generator = ScenarioLLMGenerator()
        content = '''여기 JSON입니다:
```json
{"title": "테스트"}
```
추가 텍스트'''

        result = generator._extract_json(content)
        assert result == '{"title": "테스트"}'

    def test_extract_json_from_plain_block(self):
        """일반 코드 블록에서 JSON 추출"""
        generator = ScenarioLLMGenerator()
        content = '''```
{"title": "테스트"}
```'''

        result = generator._extract_json(content)
        assert result == '{"title": "테스트"}'

    def test_extract_json_raw(self):
        """순수 JSON 추출"""
        generator = ScenarioLLMGenerator()
        content = '{"title": "테스트"}'

        result = generator._extract_json(content)
        assert result == '{"title": "테스트"}'

    def test_extract_json_with_surrounding_text(self):
        """주변 텍스트가 있는 JSON 추출"""
        generator = ScenarioLLMGenerator()
        content = '다음은 결과입니다: {"title": "테스트"} 완료.'

        result = generator._extract_json(content)
        assert '{"title": "테스트"}' in result


# =============================================================================
# Pydantic 검증 테스트
# =============================================================================

class TestPydanticValidation:
    """Pydantic 스키마 검증 테스트"""

    def test_valid_scenario_output(self, valid_llm_response):
        """유효한 시나리오 출력 검증"""
        generator = ScenarioLLMGenerator()
        json_str = generator._extract_json(valid_llm_response)
        validated = generator._validate_output(json_str)

        assert validated.title == "VR 기반 과학 실험 교육"
        assert validated.difficulty == "easy"
        assert len(validated.learning_goals) == 2

    def test_context_schema_validation(self):
        """Context 스키마 검증"""
        context = ContextSchema(
            target_audience="학생",
            prior_knowledge="없음",
            duration="1시간",
            learning_environment="교실",
            class_size=20,
        )

        assert context.target_audience == "학생"
        assert context.class_size == 20

    def test_context_schema_with_string_class_size(self):
        """문자열 class_size 허용"""
        context = ContextSchema(
            target_audience="학생",
            prior_knowledge="없음",
            duration="1시간",
            learning_environment="교실",
            class_size="20명",
        )

        assert context.class_size == "20명"

    def test_constraints_schema_validation(self):
        """Constraints 스키마 검증"""
        constraints = ConstraintsSchema(
            budget="medium",
            resources=["PC", "프로젝터"],
        )

        assert constraints.budget == "medium"
        assert len(constraints.resources) == 2

    def test_invalid_json_raises_error(self, invalid_json_response):
        """잘못된 JSON은 에러 발생"""
        generator = ScenarioLLMGenerator()

        with pytest.raises(json.JSONDecodeError):
            json_str = generator._extract_json(invalid_json_response)
            generator._validate_output(json_str)


# =============================================================================
# ScenarioSchema 변환 테스트
# =============================================================================

class TestScenarioSchemaConversion:
    """ScenarioSchema 변환 테스트"""

    def test_to_scenario_schema(self, valid_llm_response):
        """ScenarioSchema 변환"""
        generator = ScenarioLLMGenerator()
        json_str = generator._extract_json(valid_llm_response)
        validated = generator._validate_output(json_str)
        scenario = generator._to_scenario_schema(validated, "TEST-001")

        assert scenario.scenario_id == "TEST-001"
        assert scenario.title == "VR 기반 과학 실험 교육"
        assert scenario.difficulty == "easy"
        assert scenario.context["class_size"] == 20

    def test_class_size_string_conversion(self):
        """문자열 class_size를 int로 변환"""
        generator = ScenarioLLMGenerator()

        validated = LLMScenarioOutput(
            title="테스트",
            context=ContextSchema(
                target_audience="학생",
                prior_knowledge="없음",
                duration="1시간",
                learning_environment="교실",
                class_size="30명",
            ),
            learning_goals=["목표1"],
            constraints=ConstraintsSchema(budget="low", resources=[]),
            difficulty="easy",
            domain="테스트",
        )

        scenario = generator._to_scenario_schema(validated, "TEST-002")
        assert scenario.context["class_size"] == 30


# =============================================================================
# Mock LLM 테스트
# =============================================================================

class TestGeneratorWithMock:
    """Mock LLM을 사용한 통합 테스트"""

    def test_generate_success(
        self, sample_prompt_result, sample_record, valid_llm_response
    ):
        """정상 생성 테스트"""
        generator = ScenarioLLMGenerator()

        # LLM 호출 모킹
        mock_response = Mock()
        mock_response.content = valid_llm_response
        generator._llm = Mock()
        generator._llm.invoke = Mock(return_value=mock_response)

        result = generator.generate(
            prompt_result=sample_prompt_result,
            record=sample_record,
            scenario_id="IDLD-0001",
        )

        assert result.is_success
        assert result.status == GenerationStatus.SUCCESS
        assert result.scenario.scenario_id == "IDLD-0001"
        assert result.scenario.title == "VR 기반 과학 실험 교육"
        assert result.retry_count == 0

    def test_generate_with_retry(
        self, sample_prompt_result, sample_record, invalid_json_response, valid_llm_response
    ):
        """재시도 로직 테스트"""
        generator = ScenarioLLMGenerator(max_retries=3)

        # 첫 번째 호출은 실패, 두 번째는 성공
        mock_response_fail = Mock()
        mock_response_fail.content = invalid_json_response

        mock_response_success = Mock()
        mock_response_success.content = valid_llm_response

        generator._llm = Mock()
        generator._llm.invoke = Mock(
            side_effect=[mock_response_fail, mock_response_success]
        )

        # 재시도 시 새 LLM 인스턴스 생성을 모킹
        with patch(
            "shared.models.scenario_llm_generator.ChatOpenAI"
        ) as mock_chat:
            mock_chat.return_value.invoke = Mock(return_value=mock_response_success)

            result = generator.generate(
                prompt_result=sample_prompt_result,
                record=sample_record,
                scenario_id="IDLD-0002",
            )

        # 첫 번째 시도 실패 후 두 번째 시도에서 성공
        assert result.is_success or result.retry_count > 0

    def test_generate_all_retries_fail(
        self, sample_prompt_result, sample_record, invalid_json_response
    ):
        """모든 재시도 실패"""
        generator = ScenarioLLMGenerator(max_retries=2)

        mock_response = Mock()
        mock_response.content = invalid_json_response
        generator._llm = Mock()
        generator._llm.invoke = Mock(return_value=mock_response)

        with patch(
            "shared.models.scenario_llm_generator.ChatOpenAI"
        ) as mock_chat:
            mock_chat.return_value.invoke = Mock(return_value=mock_response)

            result = generator.generate(
                prompt_result=sample_prompt_result,
                record=sample_record,
                scenario_id="IDLD-0003",
            )

        assert not result.is_success
        assert result.status == GenerationStatus.PARSE_ERROR
        assert result.retry_count == 2


# =============================================================================
# 배치 생성 테스트
# =============================================================================

class TestBatchGeneration:
    """배치 생성 테스트"""

    def test_generate_batch(self, sample_prompt_result, sample_record, valid_llm_response):
        """배치 생성 테스트"""
        generator = ScenarioLLMGenerator()

        mock_response = Mock()
        mock_response.content = valid_llm_response
        generator._llm = Mock()
        generator._llm.invoke = Mock(return_value=mock_response)

        prompt_results = [sample_prompt_result, sample_prompt_result]
        records = [sample_record, sample_record]

        batch_result = generator.generate_batch(
            prompt_results=prompt_results,
            records=records,
            scenario_prefix="TEST",
            start_index=1,
        )

        assert batch_result.total_count == 2
        assert batch_result.success_count == 2
        assert batch_result.failed_count == 0
        assert len(batch_result.results) == 2

    def test_generate_batch_length_mismatch(
        self, sample_prompt_result, sample_record
    ):
        """배치 생성 길이 불일치 에러"""
        generator = ScenarioLLMGenerator()

        prompt_results = [sample_prompt_result, sample_prompt_result]
        records = [sample_record]  # 길이 불일치

        with pytest.raises(ValueError, match="길이가 일치"):
            generator.generate_batch(
                prompt_results=prompt_results,
                records=records,
            )

    def test_batch_generation_result_from_results(self):
        """BatchGenerationResult.from_results 테스트"""
        results = [
            GenerationResult(
                scenario=None,
                status=GenerationStatus.SUCCESS,
            ),
            GenerationResult(
                scenario=None,
                status=GenerationStatus.PARSE_ERROR,
            ),
            GenerationResult(
                scenario=None,
                status=GenerationStatus.SUCCESS,
            ),
        ]

        batch = BatchGenerationResult.from_results(results)

        assert batch.total_count == 3
        assert batch.success_count == 2
        assert batch.failed_count == 1


# =============================================================================
# 결과 저장 테스트
# =============================================================================

class TestResultSaving:
    """결과 저장 테스트"""

    def test_save_results(self, tmp_path, valid_llm_response):
        """결과 저장 테스트"""
        generator = ScenarioLLMGenerator()
        json_str = generator._extract_json(valid_llm_response)
        validated = generator._validate_output(json_str)
        scenario = generator._to_scenario_schema(validated, "TEST-001")

        # 배치 결과 생성
        result = GenerationResult(
            scenario=scenario,
            status=GenerationStatus.SUCCESS,
            source_record_no=1,
        )

        batch_result = BatchGenerationResult.from_results([result])
        batch_result.results[0].scenario = scenario

        # 저장
        save_stats = ScenarioLLMGenerator.save_results(
            batch_result=batch_result,
            output_dir=str(tmp_path),
        )

        assert save_stats["saved_count"] == 1
        assert save_stats["by_difficulty"]["easy"] == 1

        # 파일 확인
        saved_file = tmp_path / "easy" / "scenario_test_001.json"
        assert saved_file.exists()

        with open(saved_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["scenario_id"] == "TEST-001"

    def test_save_results_with_mapping(self, tmp_path, valid_llm_response, sample_record):
        """출처 매핑 포함 저장 테스트"""
        generator = ScenarioLLMGenerator()
        json_str = generator._extract_json(valid_llm_response)
        validated = generator._validate_output(json_str)
        scenario = generator._to_scenario_schema(validated, "TEST-002")

        result = GenerationResult(
            scenario=scenario,
            status=GenerationStatus.SUCCESS,
            source_record_no=sample_record.no,
        )

        batch_result = BatchGenerationResult.from_results([result])
        batch_result.results[0].scenario = scenario
        batch_result.source_mapping.add("TEST-002", sample_record)

        mapping_path = tmp_path / "mapping.json"

        save_stats = ScenarioLLMGenerator.save_results(
            batch_result=batch_result,
            output_dir=str(tmp_path),
            mapping_path=str(mapping_path),
        )

        assert mapping_path.exists()
        assert save_stats["mapping_path"] == str(mapping_path)
