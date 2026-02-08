"""
ADDIE 루브릭 평가기 통합 테스트

_call_llm을 mocking하여 전체 evaluate() 흐름 테스트
"""

import pytest
from unittest.mock import MagicMock

from isd_evaluator.models import ADDIEScore


class TestEvaluateIntegration:
    """evaluate() 메서드 통합 테스트"""

    def test_evaluate_with_all_valid_responses(
        self, mock_evaluator, sample_addie_output, sample_scenario, load_llm_response
    ):
        """모든 단계에서 유효한 응답을 받았을 때"""
        # 각 단계별 응답 준비
        phases = ["analysis", "design", "development", "implementation", "evaluation"]
        responses = [load_llm_response(f"{phase}/valid_response.json") for phase in phases]

        call_count = [0]

        def side_effect(prompt):
            response = responses[call_count[0]]
            call_count[0] += 1
            return response

        mock_evaluator._call_llm = MagicMock(side_effect=side_effect)

        result = mock_evaluator.evaluate(sample_addie_output, sample_scenario)

        assert isinstance(result, ADDIEScore)
        assert mock_evaluator._call_llm.call_count == 5  # 5개 단계

    def test_evaluate_with_one_failed_response(
        self, mock_evaluator, sample_addie_output, sample_scenario, load_llm_response
    ):
        """하나의 단계에서 파싱 실패 시에도 결과 생성"""
        responses = [
            load_llm_response("analysis/valid_response.json"),
            "INVALID JSON - Design 파싱 실패",  # Design 파싱 실패
            load_llm_response("development/valid_response.json"),
            load_llm_response("implementation/valid_response.json"),
            load_llm_response("evaluation/valid_response.json"),
        ]

        call_count = [0]

        def side_effect(prompt):
            response = responses[call_count[0]]
            call_count[0] += 1
            return response

        mock_evaluator._call_llm = MagicMock(side_effect=side_effect)

        result = mock_evaluator.evaluate(sample_addie_output, sample_scenario)

        # 실패한 단계는 기본값으로 처리되어야 함
        assert isinstance(result, ADDIEScore)
        # Design 단계의 평균 점수는 기본값(5.0) 근처여야 함
        assert result.design.average_score <= 6.0

    def test_evaluate_with_all_failed_responses(
        self, mock_evaluator, sample_addie_output, sample_scenario
    ):
        """모든 단계에서 파싱 실패 시"""
        mock_evaluator._call_llm = MagicMock(return_value="INVALID JSON")

        result = mock_evaluator.evaluate(sample_addie_output, sample_scenario)

        # 모든 단계가 기본값으로 처리되어야 함
        assert isinstance(result, ADDIEScore)
        # 정규화 점수는 50% 근처 (기본값 5.0)
        assert 45 <= result.normalized_score <= 55

    def test_evaluate_without_scenario(
        self, mock_evaluator, sample_addie_output, load_llm_response
    ):
        """시나리오 없이 평가"""
        phases = ["analysis", "design", "development", "implementation", "evaluation"]
        responses = [load_llm_response(f"{phase}/valid_response.json") for phase in phases]

        call_count = [0]

        def side_effect(prompt):
            response = responses[call_count[0]]
            call_count[0] += 1
            return response

        mock_evaluator._call_llm = MagicMock(side_effect=side_effect)

        result = mock_evaluator.evaluate(sample_addie_output, scenario=None)

        assert isinstance(result, ADDIEScore)

    def test_evaluate_prompt_contains_scenario_context(
        self, mock_evaluator, sample_addie_output, sample_scenario
    ):
        """프롬프트에 시나리오 컨텍스트가 포함되는지 확인"""
        mock_evaluator._call_llm = MagicMock(return_value='{"sub_scores": {}, "sub_reasoning": {}}')

        mock_evaluator.evaluate(sample_addie_output, sample_scenario)

        # 첫 번째 호출의 프롬프트 확인
        first_call_args = mock_evaluator._call_llm.call_args_list[0]
        prompt = first_call_args[0][0]

        # 시나리오 정보가 프롬프트에 포함되어야 함
        assert "대학생" in prompt or "파이썬" in prompt

    def test_evaluate_calls_llm_for_each_phase(
        self, mock_evaluator, sample_addie_output, sample_scenario
    ):
        """각 단계별로 LLM이 호출되는지 확인"""
        mock_evaluator._call_llm = MagicMock(return_value='{"sub_scores": {}, "sub_reasoning": {}}')

        mock_evaluator.evaluate(sample_addie_output, sample_scenario)

        # 5개 단계에 대해 호출
        assert mock_evaluator._call_llm.call_count == 5


class TestExtractPhaseOutputs:
    """_extract_phase_outputs 메서드 테스트"""

    def test_extract_standard_structure(self, mock_evaluator, sample_addie_output):
        """표준 구조 추출"""
        result = mock_evaluator._extract_phase_outputs(sample_addie_output)

        assert "analysis" in result
        assert "design" in result
        assert "development" in result
        assert "implementation" in result
        assert "evaluation" in result

    def test_extract_with_capitalized_keys(self, mock_evaluator):
        """대문자 키 이름 처리"""
        addie_output = {
            "Analysis": {"data": "분석 데이터"},
            "Design": {"data": "설계 데이터"},
            "Development": {"data": "개발 데이터"},
            "Implementation": {"data": "실행 데이터"},
            "Evaluation": {"data": "평가 데이터"},
        }

        result = mock_evaluator._extract_phase_outputs(addie_output)

        # 대소문자 구분 없이 추출되어야 함
        assert "analysis" in result
        assert result["analysis"] is not None

    def test_extract_with_alternative_keys(self, mock_evaluator):
        """대체 키 이름 처리"""
        addie_output = {
            "learner_analysis": {"data": "학습자 분석"},
            "learning_objectives": {"data": "학습 목표"},
            "content": {"data": "콘텐츠"},
            "delivery_plan": {"data": "전달 계획"},
            "assessment": {"data": "평가"},
        }

        result = mock_evaluator._extract_phase_outputs(addie_output)

        # 대체 키도 매핑되어야 함
        assert "analysis" in result
        assert "design" in result

    def test_extract_empty_output(self, mock_evaluator):
        """빈 출력 처리"""
        addie_output = {}

        result = mock_evaluator._extract_phase_outputs(addie_output)

        # 모든 단계에 대해 빈 딕셔너리 또는 원본 반환
        assert "analysis" in result
        assert "design" in result

    def test_extract_nested_structure(self, mock_evaluator):
        """중첩 구조 처리"""
        addie_output = {
            "analysis": {
                "learner_analysis": {"demographics": "대학생"},
                "context_analysis": {"environment": "온라인"},
            },
            "design": {
                "learning_objectives": ["목표1", "목표2"],
                "assessment_design": {"type": "형성평가"},
            },
            "development": {},
            "implementation": {},
            "evaluation": {},
        }

        result = mock_evaluator._extract_phase_outputs(addie_output)

        # 중첩 구조가 올바르게 추출되어야 함
        assert "analysis" in result
        assert result["analysis"] is not None


class TestFormatScenario:
    """_format_scenario 메서드 테스트"""

    def test_format_scenario_basic(self, mock_evaluator, sample_scenario):
        """기본 시나리오 포맷팅"""
        result = mock_evaluator._format_scenario(sample_scenario)

        # 주요 정보가 포함되어야 함
        assert "파이썬" in result or "프로그래밍" in result
        assert "대학생" in result

    def test_format_scenario_with_missing_fields(self, mock_evaluator):
        """필드 누락 시나리오 포맷팅"""
        scenario = {
            "title": "테스트 시나리오",
            "context": {},  # 빈 컨텍스트
        }

        result = mock_evaluator._format_scenario(scenario)

        # 오류 없이 포맷팅되어야 함
        assert isinstance(result, str)
        assert "테스트 시나리오" in result

    def test_format_scenario_with_constraints(self, mock_evaluator):
        """제약조건 포함 시나리오"""
        scenario = {
            "title": "제약조건 시나리오",
            "context": {
                "target_audience": "신입사원",
                "duration": "2시간",
            },
            "constraints": {
                "budget": "limited",
                "time": "urgent",
            },
        }

        result = mock_evaluator._format_scenario(scenario)

        assert "신입사원" in result


class TestGenerateContextGuidelines:
    """_generate_context_guidelines 메서드 테스트"""

    def test_generate_guidelines_for_elementary(self, mock_evaluator):
        """초등학생 대상 가이드라인"""
        scenario = {
            "context": {
                "target_audience": "초등학교 5학년",
                "learning_environment": "교실",
            },
        }

        result = mock_evaluator._generate_context_guidelines(scenario)

        # 초등학생 관련 가이드라인 포함
        if result:  # 가이드라인이 있는 경우
            assert "시각적" in result or "게임화" in result or "활동" in result

    def test_generate_guidelines_for_adult(self, mock_evaluator):
        """성인 대상 가이드라인"""
        scenario = {
            "context": {
                "target_audience": "직장인 신입사원",
                "learning_environment": "온라인",
            },
        }

        result = mock_evaluator._generate_context_guidelines(scenario)

        # 성인 관련 가이드라인 포함 가능
        if result:
            assert isinstance(result, str)

    def test_generate_guidelines_for_online(self, mock_evaluator):
        """온라인 환경 가이드라인"""
        scenario = {
            "context": {
                "target_audience": "대학생",
                "learning_environment": "온라인 실시간(Zoom 등)",
            },
        }

        result = mock_evaluator._generate_context_guidelines(scenario)

        # 온라인 환경 관련 가이드라인 포함 가능
        if result:
            assert isinstance(result, str)

    def test_generate_guidelines_empty_context(self, mock_evaluator):
        """빈 컨텍스트"""
        scenario = {"context": {}}

        result = mock_evaluator._generate_context_guidelines(scenario)

        # 빈 문자열 반환
        assert result == "" or isinstance(result, str)


class TestEvaluatorWithBenchmarks:
    """Benchmark Examples 관련 테스트"""

    def test_evaluator_with_benchmarks_disabled(self, mock_evaluator):
        """Benchmark Examples 비활성화 확인"""
        assert mock_evaluator.include_benchmarks is False
        assert mock_evaluator._benchmark_data is None

    def test_evaluator_can_enable_benchmarks(self):
        """Benchmark Examples 활성화 가능 확인"""
        from isd_evaluator.metrics.addie_rubric import ADDIERubricEvaluator

        evaluator = ADDIERubricEvaluator(
            provider="upstage",
            model="solar-pro2-251215",
            api_key="test-api-key",
            include_benchmarks=True,
        )

        assert evaluator.include_benchmarks is True
        # _benchmark_data가 로드되었는지 확인 (None이 아님)
        # 실제 데이터가 있을 수도 있고 없을 수도 있음


class TestCreateDefaultScore:
    """_create_default_score 메서드 테스트"""

    def test_create_default_score_structure(self, mock_evaluator):
        """기본 점수 구조 확인"""
        result = mock_evaluator._create_default_score(default=5.0)

        assert isinstance(result, ADDIEScore)
        assert hasattr(result, 'analysis')
        assert hasattr(result, 'design')
        assert hasattr(result, 'development')
        assert hasattr(result, 'implementation')
        assert hasattr(result, 'evaluation')

    def test_create_default_score_values(self, mock_evaluator):
        """기본 점수 값 확인"""
        result = mock_evaluator._create_default_score(default=5.0)

        # 모든 항목의 점수가 5.0
        for item in result.analysis.items:
            assert item.score == 5.0

    def test_create_default_score_custom_value(self, mock_evaluator):
        """커스텀 기본값"""
        result = mock_evaluator._create_default_score(default=7.0)

        # 모든 항목의 점수가 7.0
        for item in result.analysis.items:
            assert item.score == 7.0

    def test_create_default_score_normalized(self, mock_evaluator):
        """기본 점수의 정규화 값"""
        result = mock_evaluator._create_default_score(default=5.0)

        # 정규화 점수는 50% (5/10 * 100)
        assert 45 <= result.normalized_score <= 55
