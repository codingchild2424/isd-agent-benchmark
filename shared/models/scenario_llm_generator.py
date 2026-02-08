"""
시나리오 LLM 생성기

PromptBuildResult를 LLM에 전달하여 ScenarioSchema JSON을 생성합니다.

주요 기능:
- LLM 호출 (ChatOpenAI + Upstage API)
- JSON 블록 추출 및 파싱
- Pydantic 검증
- ScenarioSchema 변환
- 재시도 로직 및 에러 핸들링
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

from .prompt_builder import PromptBuildResult
from .idld_dataset import ScenarioSchema, IDLDRecord, SourceMapping


# =============================================================================
# 상수 및 설정
# =============================================================================

UPSTAGE_BASE_URL = "https://api.upstage.ai/v1/solar"
UPSTAGE_DEFAULT_MODEL = "solar-mini"


# =============================================================================
# Pydantic 스키마 (LLM 출력 검증용)
# =============================================================================

class ContextSchema(BaseModel):
    """시나리오 context 필드 스키마 (v3 - Context Matrix 정렬 + 선택적 개선)"""
    # 기존 필드 (deprecated, 하위 호환성용)
    target_audience: Optional[str] = None

    # 신규 필드 (Context Matrix 정렬)
    learner_age: Optional[str] = None       # CSV 1-4번: 10대, 20대, 30대, 40대 이상
    learner_education: Optional[str] = None # CSV 5-9번: 초등, 중등, 고등, 대학, 성인 학습자(비학위)
    domain_expertise: Optional[str] = None  # CSV 10-12번: 초급, 중급, 고급
    learner_role: Optional[str] = None      # CSV 13-16번: 학생, 직장인, 전문직, 예비 교사/교사

    # 기존 필드 유지
    prior_knowledge: str
    duration: str
    learning_environment: str
    class_size: Union[int, str]             # CSV 40-42번: 정수 또는 범주형 문자열
    institution_type: Optional[str] = None
    additional_context: Optional[str] = None


class ConstraintsSchema(BaseModel):
    """시나리오 constraints 필드 스키마 (v2 - Context Matrix 정렬)"""
    budget: str
    resources: List[str] = Field(default_factory=list)
    tech_requirements: Optional[str] = None
    accessibility: Optional[Union[List[str], str]] = None
    language: str = "ko"

    # 신규 필드 (Context Matrix 정렬)
    assessment_type: Optional[str] = None   # CSV 43-45번: 형성평가/총괄평가/프로젝트 기반 평가


class LLMScenarioOutput(BaseModel):
    """LLM이 생성하는 시나리오 JSON 스키마"""
    title: str
    context: ContextSchema
    learning_goals: List[str] = Field(min_length=1)
    constraints: ConstraintsSchema
    difficulty: str
    domain: str


# =============================================================================
# 생성 상태 및 결과
# =============================================================================

class GenerationStatus(str, Enum):
    """생성 상태"""
    SUCCESS = "success"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    API_ERROR = "api_error"


@dataclass
class GenerationResult:
    """단일 시나리오 생성 결과"""
    scenario: Optional[ScenarioSchema]
    status: GenerationStatus
    source_record_no: int = 0
    raw_response: str = ""
    error_message: str = ""
    retry_count: int = 0
    generation_time: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.status == GenerationStatus.SUCCESS


@dataclass
class BatchGenerationResult:
    """배치 생성 결과"""
    results: List[GenerationResult]
    total_count: int
    success_count: int
    failed_count: int
    source_mapping: SourceMapping
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_results(cls, results: List[GenerationResult]) -> "BatchGenerationResult":
        success = sum(1 for r in results if r.is_success)
        return cls(
            results=results,
            total_count=len(results),
            success_count=success,
            failed_count=len(results) - success,
            source_mapping=SourceMapping(),
        )


# =============================================================================
# ScenarioLLMGenerator 클래스
# =============================================================================

class ScenarioLLMGenerator:
    """
    프롬프트를 LLM에 전달하여 시나리오 JSON을 생성합니다.

    사용 예:
        generator = ScenarioLLMGenerator()

        # 단일 생성
        result = generator.generate(prompt_result, record, scenario_id="IDLD-0001")
        if result.is_success:
            print(result.scenario.to_json())

        # 배치 생성
        batch_result = generator.generate_batch(prompts, records)
    """

    # 재시도 설정
    MAX_RETRIES = 3
    RETRY_TEMPERATURE_INCREMENT = 0.1

    def __init__(
        self,
        model: str = UPSTAGE_DEFAULT_MODEL,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Args:
            model: LLM 모델명
            temperature: 샘플링 온도
            api_key: Upstage API 키 (없으면 환경변수에서 로드)
            max_retries: 최대 재시도 횟수
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.max_retries = max_retries
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        """LLM 인스턴스 (지연 초기화)"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
                base_url=UPSTAGE_BASE_URL,
            )
        return self._llm

    # =========================================================================
    # 메인 API
    # =========================================================================

    def generate(
        self,
        prompt_result: PromptBuildResult,
        record: IDLDRecord,
        scenario_id: str,
        variant_type: str = "realistic",
    ) -> GenerationResult:
        """
        단일 시나리오 생성

        Args:
            prompt_result: PromptBuilder.build() 결과
            record: 원본 IDLD 레코드 (출처 매핑용)
            scenario_id: 시나리오 ID (예: "IDLD-0001")
            variant_type: 변형 유형 ("realistic" | "challenging")

        Returns:
            GenerationResult
        """
        start_time = datetime.now()
        raw_response = ""
        last_error = ""

        for retry in range(self.max_retries):
            try:
                # 1. LLM 호출
                raw_response = self._call_llm(prompt_result.prompt, retry)

                # 2. JSON 추출
                json_str = self._extract_json(raw_response)

                # 3. Pydantic 검증
                validated = self._validate_output(json_str)

                # 4. ScenarioSchema 변환 (variant_type 포함)
                scenario = self._to_scenario_schema(validated, scenario_id, variant_type)

                elapsed = (datetime.now() - start_time).total_seconds()
                return GenerationResult(
                    scenario=scenario,
                    status=GenerationStatus.SUCCESS,
                    source_record_no=record.no,
                    raw_response=raw_response,
                    retry_count=retry,
                    generation_time=elapsed,
                )

            except json.JSONDecodeError as e:
                last_error = f"JSON 파싱 오류: {str(e)}"
                continue

            except ValidationError as e:
                last_error = f"검증 오류: {str(e)}"
                continue

            except Exception as e:
                last_error = f"API 오류: {str(e)}"
                continue

        # 모든 재시도 실패
        elapsed = (datetime.now() - start_time).total_seconds()
        return GenerationResult(
            scenario=None,
            status=GenerationStatus.PARSE_ERROR,
            source_record_no=record.no,
            raw_response=raw_response,
            error_message=last_error,
            retry_count=self.max_retries,
            generation_time=elapsed,
        )

    def generate_batch(
        self,
        prompt_results: List[PromptBuildResult],
        records: List[IDLDRecord],
        scenario_prefix: str = "IDLD",
        start_index: int = 1,
        variant_types: Optional[List[str]] = None,
        progress_callback=None,
    ) -> BatchGenerationResult:
        """
        배치 시나리오 생성

        Args:
            prompt_results: PromptBuilder 결과 리스트
            records: IDLD 레코드 리스트 (1:1 매핑)
            scenario_prefix: ID 접두사 (기본: "IDLD")
            start_index: 시작 인덱스
            variant_types: 변형 유형 리스트 (1:1 매핑, None이면 모두 "realistic")
            progress_callback: 진행률 콜백 함수 (i, total, result)

        Returns:
            BatchGenerationResult
        """
        if len(prompt_results) != len(records):
            raise ValueError("prompt_results와 records 길이가 일치해야 합니다")

        # variant_types 기본값 처리
        if variant_types is None:
            variant_types = ["realistic"] * len(prompt_results)
        elif len(variant_types) != len(prompt_results):
            raise ValueError("variant_types 길이가 prompt_results와 일치해야 합니다")

        results = []
        source_mapping = SourceMapping()

        for i, (prompt_result, record, variant_type) in enumerate(
            zip(prompt_results, records, variant_types)
        ):
            scenario_id = f"{scenario_prefix}-{start_index + i:04d}"

            result = self.generate(prompt_result, record, scenario_id, variant_type)
            results.append(result)

            # 성공 시 출처 매핑 추가
            if result.is_success:
                source_mapping.add(scenario_id, record)

            # 진행률 콜백
            if progress_callback:
                progress_callback(i + 1, len(prompt_results), result)

        batch_result = BatchGenerationResult.from_results(results)
        batch_result.source_mapping = source_mapping
        return batch_result

    # =========================================================================
    # LLM 호출
    # =========================================================================

    def _call_llm(self, prompt: str, retry: int = 0) -> str:
        """LLM API 호출"""
        # 재시도 시 temperature 약간 증가
        if retry > 0:
            current_temp = min(1.0, self.temperature + retry * self.RETRY_TEMPERATURE_INCREMENT)
            llm = ChatOpenAI(
                model=self.model,
                temperature=current_temp,
                api_key=self.api_key,
                base_url=UPSTAGE_BASE_URL,
            )
        else:
            llm = self.llm

        # System 메시지
        system_msg = """당신은 교수설계 시나리오를 JSON 형식으로 생성하는 전문가입니다.
반드시 유효한 JSON만 출력하세요. 마크다운 코드 블록(```json)으로 감싸서 응답하세요."""

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        return response.content

    # =========================================================================
    # JSON 파싱 및 검증
    # =========================================================================

    def _extract_json(self, content: str) -> str:
        """LLM 응답에서 JSON 블록 추출"""
        # ```json ... ``` 패턴
        if "```json" in content:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
            if match:
                return match.group(1).strip()

        # ``` ... ``` 패턴
        if "```" in content:
            match = re.search(r"```\s*([\s\S]*?)\s*```", content)
            if match:
                return match.group(1).strip()

        # { ... } 직접 찾기
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            return match.group(0).strip()

        return content.strip()

    def _validate_output(self, json_str: str) -> LLMScenarioOutput:
        """JSON을 Pydantic 모델로 검증"""
        data = json.loads(json_str)
        return LLMScenarioOutput(**data)

    def _to_scenario_schema(
        self,
        validated: LLMScenarioOutput,
        scenario_id: str,
        variant_type: str = "realistic",
    ) -> ScenarioSchema:
        """Pydantic 모델을 ScenarioSchema로 변환"""
        # class_size 처리 (int 또는 문자열)
        class_size = validated.context.class_size
        if isinstance(class_size, str):
            # "20명" → 20
            match = re.search(r"\d+", class_size)
            class_size = int(match.group()) if match else None

        context = {
            "target_audience": validated.context.target_audience,
            "prior_knowledge": validated.context.prior_knowledge,
            "duration": validated.context.duration,
            "learning_environment": validated.context.learning_environment,
            "class_size": class_size,
            "additional_context": validated.context.additional_context or "",
        }

        # institution_type이 있으면 추가
        if validated.context.institution_type:
            context["institution_type"] = validated.context.institution_type

        constraints = {
            "budget": validated.constraints.budget,
            "resources": validated.constraints.resources,
            "accessibility": validated.constraints.accessibility,
            "language": validated.constraints.language,
        }

        if validated.constraints.tech_requirements:
            constraints["tech_requirements"] = validated.constraints.tech_requirements

        return ScenarioSchema(
            scenario_id=scenario_id,
            title=validated.title,
            context=context,
            learning_goals=validated.learning_goals,
            constraints=constraints,
            difficulty=validated.difficulty,
            domain=validated.domain,
            variant_type=variant_type,
        )

    # =========================================================================
    # 저장 유틸리티
    # =========================================================================

    @staticmethod
    def save_results(
        batch_result: BatchGenerationResult,
        output_dir: str,
        mapping_path: Optional[str] = None,
    ) -> Dict:
        """
        생성 결과 저장

        Args:
            batch_result: 배치 생성 결과
            output_dir: 출력 디렉토리 (variant_type별 하위 폴더 자동 생성)
            mapping_path: 출처 매핑 파일 경로

        Returns:
            저장 통계 딕셔너리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, List[str]] = {"idld_aligned": [], "context_variant": []}

        for result in batch_result.results:
            if not result.is_success or not result.scenario:
                continue

            scenario = result.scenario
            variant_type = scenario.variant_type

            # variant_type별 폴더 생성
            type_dir = output_path / variant_type
            type_dir.mkdir(parents=True, exist_ok=True)

            # 파일 저장
            filename = f"scenario_{scenario.scenario_id.lower().replace('-', '_')}.json"
            filepath = type_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(scenario.to_json())

            if variant_type in saved_files:
                saved_files[variant_type].append(str(filepath))
            else:
                saved_files[variant_type] = [str(filepath)]

        # 출처 매핑 저장
        if mapping_path:
            # 매핑 파일 디렉토리 생성
            mapping_dir = Path(mapping_path).parent
            mapping_dir.mkdir(parents=True, exist_ok=True)
            batch_result.source_mapping.save(mapping_path)

        return {
            "output_dir": str(output_dir),
            "saved_count": batch_result.success_count,
            "by_variant_type": {k: len(v) for k, v in saved_files.items()},
            "files": saved_files,
            "mapping_path": mapping_path,
        }
