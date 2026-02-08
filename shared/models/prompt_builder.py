"""
프롬프트 빌더 (Prompt Builder)

시드(Seed), 컨텍스트(Context), IDLD 레코드를 결합하여
LLM에게 주입할 시나리오 생성 프롬프트를 자동 생성합니다.

주요 기능:
- 다국어 지원 (한국어/영어)
- 변수 슬롯 자동 주입
- 프롬프트 길이 검증
- 데이터 호환성 유지 (Evaluator와 일치)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .seed_extractor import ScenarioSeed
from .context_matrix import ContextCombination
from .idld_dataset import IDLDRecord


# =============================================================================
# 언어 설정
# =============================================================================

class Language(str, Enum):
    """지원 언어"""
    KO = "ko"
    EN = "en"


# =============================================================================
# 프롬프트 템플릿
# =============================================================================

# 한국어 템플릿
TEMPLATE_KO = """당신은 경력 10년 이상의 교수설계 전문가입니다.
다음 정보를 바탕으로 교수설계 요청 시나리오를 작성해주세요.

## 핵심 주제
{topic}

## 교수법/학습 방법
{pedagogical_method}

## 참고 연구 내용 (직접 인용하지 마세요)
{source_abstract}

## 교육 맥락
- 대상 학습자: {learner_age}, {learner_education}
- 학습자 역할: {learner_role}
- 도메인 전문성: {domain_expertise}
- 기관 유형: {institution_type}
- 교과/직무 분야: {education_domain}
- 전달 방식: {delivery_mode}
- 학습자 규모: {class_size}
- 평가 방식: {evaluation_focus}
- 기술 환경: {tech_environment}
- 교육 기간: {duration}

## 난이도
{difficulty}

## 출력 형식 (JSON)
다음 형식의 JSON으로 응답해주세요:
```json
{{
  "title": "교육 과정 제목",
  "context": {{
    "target_audience": "{learner_age} {learner_education}",
    "prior_knowledge": "필요한 사전 지식",
    "duration": "{duration}",
    "learning_environment": "{delivery_mode}",
    "class_size": "{class_size}",
    "institution_type": "{institution_type}",
    "additional_context": "추가 맥락 설명"
  }},
  "learning_goals": ["학습목표1", "학습목표2", "학습목표3"],
  "constraints": {{
    "budget": "low/medium/high",
    "resources": ["필요한 리소스"],
    "tech_requirements": "{tech_environment}",
    "accessibility": null,
    "language": "ko"
  }},
  "difficulty": "{difficulty}",
  "domain": "{education_domain}"
}}
```

## 중요 지침
1. 원본 논문의 제목, 저자, DOI 등을 절대 포함하지 마세요
2. 참고 연구 내용을 일반적인 교육 상황으로 변환하세요
3. context 필드의 값들은 위에 제공된 교육 맥락 값과 **정확히 일치**시키세요
4. 한국어로 작성하세요
5. 학습목표는 구체적이고 측정 가능하게 작성하세요
"""

# 영어 템플릿
TEMPLATE_EN = """You are an instructional design expert with over 10 years of experience.
Based on the following information, create an instructional design request scenario.

## Core Topic
{topic}

## Pedagogical Method
{pedagogical_method}

## Reference Research (Do not quote directly)
{source_abstract}

## Educational Context
- Target Learners: {learner_age}, {learner_education}
- Learner Role: {learner_role}
- Domain Expertise: {domain_expertise}
- Institution Type: {institution_type}
- Subject/Job Domain: {education_domain}
- Delivery Mode: {delivery_mode}
- Class Size: {class_size}
- Evaluation Focus: {evaluation_focus}
- Technology Environment: {tech_environment}
- Duration: {duration}

## Difficulty Level
{difficulty}

## Output Format (JSON)
Please respond with JSON in the following format:
```json
{{
  "title": "Course Title",
  "context": {{
    "target_audience": "{learner_age} {learner_education}",
    "prior_knowledge": "Required prior knowledge",
    "duration": "{duration}",
    "learning_environment": "{delivery_mode}",
    "class_size": "{class_size}",
    "institution_type": "{institution_type}",
    "additional_context": "Additional context description"
  }},
  "learning_goals": ["Learning Goal 1", "Learning Goal 2", "Learning Goal 3"],
  "constraints": {{
    "budget": "low/medium/high",
    "resources": ["Required resources"],
    "tech_requirements": "{tech_environment}",
    "accessibility": null,
    "language": "en"
  }},
  "difficulty": "{difficulty}",
  "domain": "{education_domain}"
}}
```

## Important Instructions
1. Never include the original paper's title, authors, or DOI
2. Transform the reference research into a general educational scenario
3. Ensure context field values **exactly match** the provided educational context values above
4. Write in English
5. Make learning goals specific and measurable
"""

# 템플릿 딕셔너리
TEMPLATES: Dict[str, str] = {
    "ko": TEMPLATE_KO,
    "en": TEMPLATE_EN,
}

# 난이도 설명
DIFFICULTY_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "ko": {
        "easy": "쉬움 - 명확한 요구사항, 단순한 제약조건",
        "medium": "보통 - 일반적인 복잡도, 몇 가지 제약조건",
        "hard": "어려움 - 복잡한 요구사항, 다수의 제약조건, 모호한 요소",
    },
    "en": {
        "easy": "Easy - Clear requirements, simple constraints",
        "medium": "Medium - Normal complexity, some constraints",
        "hard": "Hard - Complex requirements, multiple constraints, ambiguous elements",
    },
}


# =============================================================================
# 프롬프트 빌더 결과
# =============================================================================

@dataclass
class PromptBuildResult:
    """프롬프트 빌드 결과"""
    prompt: str
    language: str
    variables: Dict[str, str]
    token_estimate: int
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """유효성 검사"""
        return len(self.prompt) > 0 and len(self.warnings) == 0


# =============================================================================
# PromptBuilder 클래스
# =============================================================================

class PromptBuilder:
    """
    프롬프트 빌더

    시드, 컨텍스트, IDLD 레코드를 결합하여
    완성된 LLM 프롬프트를 생성합니다.

    사용 예:
        builder = PromptBuilder()

        result = builder.build(
            seed=seed,
            context=context,
            record=record,
            difficulty="medium",
            language="ko"
        )

        print(result.prompt)
    """

    # 토큰 추정 상수 (한글 기준: ~2자/토큰, 영어: ~4자/토큰)
    CHARS_PER_TOKEN_KO = 2.0
    CHARS_PER_TOKEN_EN = 4.0

    # 기본 제한값
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_MAX_ABSTRACT_LENGTH = 1500

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_abstract_length: int = DEFAULT_MAX_ABSTRACT_LENGTH,
    ):
        """
        Args:
            max_tokens: 최대 토큰 수
            max_abstract_length: 초록 최대 길이 (잘라내기 적용)
        """
        self.max_tokens = max_tokens
        self.max_abstract_length = max_abstract_length
        self.templates = TEMPLATES.copy()

    # =========================================================================
    # 메인 API
    # =========================================================================

    def build(
        self,
        seed: ScenarioSeed,
        context: ContextCombination,
        record: IDLDRecord,
        difficulty: str = "medium",
        language: str = "ko",
    ) -> PromptBuildResult:
        """
        프롬프트 생성

        Args:
            seed: 시나리오 시드
            context: 컨텍스트 조합
            record: IDLD 레코드
            difficulty: 난이도 (easy/medium/hard)
            language: 언어 (ko/en)

        Returns:
            PromptBuildResult
        """
        warnings = []

        # 1. 변수 추출
        variables = self._extract_variables(
            seed=seed,
            context=context,
            record=record,
            difficulty=difficulty,
            language=language,
        )

        # 2. 초록 길이 검증 및 잘라내기
        abstract = variables.get("source_abstract", "")
        if len(abstract) > self.max_abstract_length:
            variables["source_abstract"] = self._truncate_abstract(
                abstract, self.max_abstract_length
            )
            warnings.append(
                f"초록이 너무 깁니다. {self.max_abstract_length}자로 잘랐습니다."
            )

        # 3. 템플릿 선택 및 변수 주입
        template = self.templates.get(language, self.templates["ko"])
        prompt = self._inject_variables(template, variables)

        # 4. 토큰 추정
        token_estimate = self._estimate_tokens(prompt, language)

        # 5. 토큰 제한 검증
        if token_estimate > self.max_tokens:
            warnings.append(
                f"예상 토큰 수({token_estimate})가 제한({self.max_tokens})을 초과합니다."
            )

        return PromptBuildResult(
            prompt=prompt,
            language=language,
            variables=variables,
            token_estimate=token_estimate,
            warnings=warnings,
        )

    def build_batch(
        self,
        seeds: List[ScenarioSeed],
        contexts: List[ContextCombination],
        records: List[IDLDRecord],
        difficulties: Optional[List[str]] = None,
        language: str = "ko",
    ) -> List[PromptBuildResult]:
        """
        배치 프롬프트 생성

        Args:
            seeds: 시드 리스트
            contexts: 컨텍스트 리스트
            records: 레코드 리스트
            difficulties: 난이도 리스트 (None이면 모두 medium)
            language: 언어

        Returns:
            PromptBuildResult 리스트
        """
        n = len(seeds)
        if len(contexts) != n or len(records) != n:
            raise ValueError("seeds, contexts, records의 길이가 일치해야 합니다")

        if difficulties is None:
            difficulties = ["medium"] * n

        results = []
        for seed, context, record, difficulty in zip(
            seeds, contexts, records, difficulties
        ):
            result = self.build(
                seed=seed,
                context=context,
                record=record,
                difficulty=difficulty,
                language=language,
            )
            results.append(result)

        return results

    # =========================================================================
    # 변수 추출
    # =========================================================================

    def _extract_variables(
        self,
        seed: ScenarioSeed,
        context: ContextCombination,
        record: IDLDRecord,
        difficulty: str,
        language: str,
    ) -> Dict[str, str]:
        """시드, 컨텍스트, 레코드에서 변수 추출"""

        # 난이도 설명
        difficulty_desc = DIFFICULTY_DESCRIPTIONS.get(language, DIFFICULTY_DESCRIPTIONS["ko"])
        difficulty_text = difficulty_desc.get(difficulty, difficulty)

        return {
            # 시드에서 추출
            "topic": seed.topic,
            "pedagogical_method": seed.pedagogical_method,

            # 레코드에서 추출
            "source_abstract": record.abstract,
            "source_keywords": ", ".join(record.keywords) if record.keywords else "",

            # 컨텍스트에서 추출 (None 값 처리)
            "learner_age": context.learner_age or "미지정",
            "learner_education": context.learner_education or "미지정",
            "domain_expertise": context.domain_expertise or "미지정",
            "learner_role": context.learner_role or "미지정",
            "institution_type": context.institution_type or "미지정",
            "education_domain": context.education_domain or "미지정",
            "delivery_mode": context.delivery_mode or "미지정",
            "class_size": context.class_size or "미지정",
            "evaluation_focus": context.evaluation_focus or "미지정",
            "tech_environment": context.tech_environment or "미지정",
            "duration": context.duration or "미지정",

            # 난이도
            "difficulty": difficulty_text,
        }

    def _inject_variables(
        self,
        template: str,
        variables: Dict[str, str],
    ) -> str:
        """템플릿에 변수 주입"""
        prompt = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            prompt = prompt.replace(placeholder, str(value))
        return prompt

    # =========================================================================
    # 검증 및 유틸리티
    # =========================================================================

    def _truncate_abstract(self, abstract: str, max_length: int) -> str:
        """초록 잘라내기 (문장 단위로 자르기 시도)"""
        if len(abstract) <= max_length:
            return abstract

        # 문장 단위로 자르기
        truncated = abstract[:max_length]

        # 마지막 문장 완성 시도
        last_period = truncated.rfind(".")
        if last_period > max_length * 0.7:  # 70% 이상 지점에 마침표가 있으면
            truncated = truncated[:last_period + 1]
        else:
            truncated = truncated.rstrip() + "..."

        return truncated

    def _estimate_tokens(self, text: str, language: str) -> int:
        """토큰 수 추정"""
        if language == "en":
            chars_per_token = self.CHARS_PER_TOKEN_EN
        else:
            chars_per_token = self.CHARS_PER_TOKEN_KO

        return int(len(text) / chars_per_token)

    def validate_prompt_length(
        self,
        prompt: str,
        language: str = "ko",
    ) -> Tuple[bool, int, str]:
        """
        프롬프트 길이 검증

        Args:
            prompt: 프롬프트 텍스트
            language: 언어

        Returns:
            (유효 여부, 예상 토큰 수, 메시지)
        """
        token_estimate = self._estimate_tokens(prompt, language)

        if token_estimate <= self.max_tokens:
            return True, token_estimate, "OK"
        else:
            return False, token_estimate, f"토큰 제한 초과: {token_estimate}/{self.max_tokens}"

    # =========================================================================
    # 템플릿 관리
    # =========================================================================

    def set_template(self, language: str, template: str) -> None:
        """커스텀 템플릿 설정"""
        self.templates[language] = template

    def get_template(self, language: str) -> str:
        """템플릿 조회"""
        return self.templates.get(language, self.templates["ko"])

    def list_languages(self) -> List[str]:
        """지원 언어 목록"""
        return list(self.templates.keys())

    # =========================================================================
    # 미리보기
    # =========================================================================

    def preview_variables(
        self,
        seed: ScenarioSeed,
        context: ContextCombination,
        record: IDLDRecord,
        difficulty: str = "medium",
        language: str = "ko",
    ) -> Dict[str, str]:
        """
        주입될 변수 미리보기 (디버깅용)

        Returns:
            변수 딕셔너리
        """
        return self._extract_variables(
            seed=seed,
            context=context,
            record=record,
            difficulty=difficulty,
            language=language,
        )

    def explain_template(self, language: str = "ko") -> str:
        """템플릿 구조 설명"""
        template = self.templates.get(language, self.templates["ko"])

        # 변수 슬롯 추출
        import re
        slots = re.findall(r"\{(\w+)\}", template)
        unique_slots = list(dict.fromkeys(slots))

        lines = [f"## 프롬프트 템플릿 ({language})"]
        lines.append(f"")
        lines.append(f"### 변수 슬롯 ({len(unique_slots)}개)")
        for slot in unique_slots:
            lines.append(f"- `{{{slot}}}`")

        lines.append(f"")
        lines.append(f"### 템플릿 길이")
        lines.append(f"- 문자 수: {len(template)}")
        lines.append(f"- 예상 토큰: ~{self._estimate_tokens(template, language)}")

        return "\n".join(lines)
