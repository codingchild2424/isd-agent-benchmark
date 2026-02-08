"""
시드 추출기 (Seed Extractor)

IDLD 논문 초록에서 교수설계 시나리오 생성을 위한
핵심 정보(Seed)를 추출합니다.

추출 항목:
- Topic: 핵심 주제 (예: "머신러닝", "생화학 대사 작용")
- Pedagogical Method: 교수법 (예: "문제 중심 학습", "VR 기반 학습")
- Categories: 교육 분야 카테고리 (예: "Higher Education", "STEM")
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .idld_dataset import IDLDRecord


# =============================================================================
# 카테고리 정의
# =============================================================================

class EducationLevel(str, Enum):
    """교육 수준 카테고리"""
    K12 = "K-12"
    HIGHER_EDUCATION = "Higher Education"
    GRADUATE = "Graduate"
    PROFESSIONAL = "Professional Development"
    CORPORATE = "Corporate Training"
    LIFELONG = "Lifelong Learning"


class SubjectDomain(str, Enum):
    """교과/분야 카테고리"""
    STEM = "STEM"
    HUMANITIES = "Humanities"
    SOCIAL_SCIENCES = "Social Sciences"
    BUSINESS = "Business"
    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"
    ARTS = "Arts"
    LANGUAGE = "Language"
    IT = "IT/Computer Science"
    OTHER = "Other"


# 키워드 → 카테고리 매핑 테이블
KEYWORD_TO_LEVEL: Dict[str, EducationLevel] = {
    # K-12
    "k-12": EducationLevel.K12,
    "primary": EducationLevel.K12,
    "elementary": EducationLevel.K12,
    "secondary": EducationLevel.K12,
    "middle school": EducationLevel.K12,
    "high school": EducationLevel.K12,
    # Higher Education
    "higher education": EducationLevel.HIGHER_EDUCATION,
    "university": EducationLevel.HIGHER_EDUCATION,
    "undergraduate": EducationLevel.HIGHER_EDUCATION,
    "college": EducationLevel.HIGHER_EDUCATION,
    # Graduate
    "graduate": EducationLevel.GRADUATE,
    "postgraduate": EducationLevel.GRADUATE,
    "doctoral": EducationLevel.GRADUATE,
    "phd": EducationLevel.GRADUATE,
    "master": EducationLevel.GRADUATE,
    # Professional
    "professional development": EducationLevel.PROFESSIONAL,
    "teacher training": EducationLevel.PROFESSIONAL,
    "faculty development": EducationLevel.PROFESSIONAL,
    # Corporate
    "corporate": EducationLevel.CORPORATE,
    "workplace": EducationLevel.CORPORATE,
    "employee training": EducationLevel.CORPORATE,
    "organizational": EducationLevel.CORPORATE,
}

KEYWORD_TO_DOMAIN: Dict[str, SubjectDomain] = {
    # STEM
    "stem": SubjectDomain.STEM,
    "science": SubjectDomain.STEM,
    "mathematics": SubjectDomain.STEM,
    "engineering": SubjectDomain.STEM,
    "physics": SubjectDomain.STEM,
    "chemistry": SubjectDomain.STEM,
    "biology": SubjectDomain.STEM,
    # IT
    "programming": SubjectDomain.IT,
    "computer science": SubjectDomain.IT,
    "software": SubjectDomain.IT,
    "coding": SubjectDomain.IT,
    "machine learning": SubjectDomain.IT,
    "artificial intelligence": SubjectDomain.IT,
    "data science": SubjectDomain.IT,
    # Healthcare
    "medical": SubjectDomain.HEALTHCARE,
    "nursing": SubjectDomain.HEALTHCARE,
    "healthcare": SubjectDomain.HEALTHCARE,
    "clinical": SubjectDomain.HEALTHCARE,
    "pharmacy": SubjectDomain.HEALTHCARE,
    # Business
    "business": SubjectDomain.BUSINESS,
    "management": SubjectDomain.BUSINESS,
    "marketing": SubjectDomain.BUSINESS,
    "finance": SubjectDomain.BUSINESS,
    "economics": SubjectDomain.BUSINESS,
    # Language
    "language learning": SubjectDomain.LANGUAGE,
    "esl": SubjectDomain.LANGUAGE,
    "efl": SubjectDomain.LANGUAGE,
    "foreign language": SubjectDomain.LANGUAGE,
    "english": SubjectDomain.LANGUAGE,
    # Education
    "teacher education": SubjectDomain.EDUCATION,
    "instructional design": SubjectDomain.EDUCATION,
    "pedagogy": SubjectDomain.EDUCATION,
    "curriculum": SubjectDomain.EDUCATION,
}


# =============================================================================
# 데이터 구조
# =============================================================================

class ExtractionStatus(str, Enum):
    """추출 상태"""
    SUCCESS = "success"
    NEEDS_REVIEW = "needs_review"  # 검토 필요 (경고 플래그)
    FAILED = "failed"


@dataclass
class ScenarioSeed:
    """시나리오 생성을 위한 시드 데이터"""
    topic: str                           # 핵심 주제
    pedagogical_method: str              # 교수법
    categories: List[str] = field(default_factory=list)  # 카테고리

    # 원본 매핑 (시나리오에 포함 안 함)
    source_record_no: int = 0

    # 메타데이터
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    warnings: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """딕셔너리 변환 (저장용)"""
        return {
            "topic": self.topic,
            "pedagogical_method": self.pedagogical_method,
            "categories": self.categories,
            "source_record_no": self.source_record_no,
            "status": self.status.value,
            "warnings": self.warnings,
            "confidence_score": self.confidence_score,
            "extracted_at": self.extracted_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ScenarioSeed":
        """딕셔너리에서 로드"""
        return cls(
            topic=data["topic"],
            pedagogical_method=data["pedagogical_method"],
            categories=data.get("categories", []),
            source_record_no=data.get("source_record_no", 0),
            status=ExtractionStatus(data.get("status", "success")),
            warnings=data.get("warnings", []),
            confidence_score=data.get("confidence_score", 1.0),
            extracted_at=data.get("extracted_at", ""),
        )

    def is_valid(self) -> bool:
        """유효성 검사"""
        return (
            self.status == ExtractionStatus.SUCCESS
            and len(self.topic) > 0
            and len(self.pedagogical_method) > 0
        )

    def needs_review(self) -> bool:
        """수동 검토 필요 여부"""
        return self.status == ExtractionStatus.NEEDS_REVIEW


# LLM 출력용 Pydantic 모델
class LLMExtractionResult(BaseModel):
    """LLM 추출 결과 스키마"""
    topic: str = Field(description="핵심 주제 (3~20자)")
    topic_english: str = Field(description="핵심 주제 영문 (3~50자)")
    pedagogical_method: str = Field(description="교수법/학습 방법 (3~30자)")
    pedagogical_method_english: str = Field(description="교수법 영문 (3~50자)")
    confidence: float = Field(
        description="추출 신뢰도 (0.0~1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="추출 근거 요약")


# =============================================================================
# SeedExtractor 클래스
# =============================================================================

# Upstage API 설정
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1/solar"
UPSTAGE_DEFAULT_MODEL = "solar-mini"


class SeedExtractor:
    """
    IDLD 초록에서 시나리오 시드를 추출하는 클래스

    사용 예:
        extractor = SeedExtractor()
        seed = extractor.extract(idld_record)

        # 배치 추출
        seeds = extractor.extract_batch(records)
    """

    # 검증 임계값
    MIN_TOPIC_LENGTH = 2
    MAX_TOPIC_LENGTH = 50
    MIN_METHOD_LENGTH = 2
    MAX_METHOD_LENGTH = 60
    MIN_CONFIDENCE = 0.6  # 이 이하면 NEEDS_REVIEW

    def __init__(
        self,
        model: str = UPSTAGE_DEFAULT_MODEL,
        temperature: float = 0.3,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            model: 사용할 LLM 모델
            temperature: 생성 온도 (낮을수록 일관성)
            api_key: Upstage API 키 (환경변수에서 자동 로드)
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
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
    # 메인 추출 메서드
    # =========================================================================

    def extract(self, record: IDLDRecord) -> ScenarioSeed:
        """
        IDLD 레코드에서 시나리오 시드 추출

        Args:
            record: IDLD 레코드

        Returns:
            ScenarioSeed 객체
        """
        # 1. LLM으로 Topic/Method 추출
        llm_result = self._extract_with_llm(record.abstract)

        # 2. 키워드 기반 카테고리 분류
        categories = self._classify_categories(record.keywords, record.abstract)

        # 3. 검증 및 Seed 생성
        seed = self._create_seed(
            llm_result=llm_result,
            categories=categories,
            source_record_no=record.no,
        )

        return seed

    def extract_batch(
        self,
        records: List[IDLDRecord],
        skip_on_error: bool = True,
    ) -> List[ScenarioSeed]:
        """
        여러 레코드에서 배치 추출

        Args:
            records: IDLD 레코드 리스트
            skip_on_error: 에러 발생 시 건너뛰기

        Returns:
            ScenarioSeed 리스트
        """
        seeds = []
        for record in records:
            try:
                seed = self.extract(record)
                seeds.append(seed)
            except Exception as e:
                if skip_on_error:
                    # 실패한 레코드는 FAILED 상태로 추가
                    failed_seed = ScenarioSeed(
                        topic="",
                        pedagogical_method="",
                        source_record_no=record.no,
                        status=ExtractionStatus.FAILED,
                        warnings=[f"추출 실패: {str(e)}"],
                        confidence_score=0.0,
                    )
                    seeds.append(failed_seed)
                else:
                    raise

        return seeds

    # =========================================================================
    # LLM 추출
    # =========================================================================

    def _extract_with_llm(self, abstract: str) -> LLMExtractionResult:
        """LLM을 사용하여 초록에서 정보 추출"""

        system_prompt = """당신은 교수설계(Instructional Design) 전문가입니다.
주어진 논문 초록에서 다음 정보를 추출해주세요:

1. **핵심 주제(Topic)**: 이 논문이 다루는 교육/학습의 핵심 주제
   - 예: "생화학 대사 작용", "프로그래밍 기초", "간호 시뮬레이션"

2. **교수법(Pedagogical Method)**: 사용된 교수/학습 방법
   - 예: "문제 중심 학습(PBL)", "VR 기반 학습", "플립드 러닝", "게이미피케이션"

중요:
- 한국어와 영문 모두 추출해주세요
- 주제와 교수법은 간결하게 (3~30자)
- 초록에서 명확히 드러나지 않으면 신뢰도를 낮게 설정하세요

응답은 반드시 JSON 형식으로:
{
  "topic": "핵심 주제 (한국어)",
  "topic_english": "Core Topic (English)",
  "pedagogical_method": "교수법 (한국어)",
  "pedagogical_method_english": "Pedagogical Method (English)",
  "confidence": 0.0~1.0,
  "reasoning": "추출 근거 요약"
}"""

        user_prompt = f"""다음 논문 초록에서 핵심 주제와 교수법을 추출해주세요:

---
{abstract}
---

JSON 형식으로 응답해주세요."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)

        # JSON 파싱
        try:
            # 응답에서 JSON 추출
            content = response.content
            # JSON 블록 추출 (```json ... ``` 형태 대응)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())
            return LLMExtractionResult(**data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 파싱 실패 시 기본값 반환
            return LLMExtractionResult(
                topic="추출 실패",
                topic_english="Extraction Failed",
                pedagogical_method="추출 실패",
                pedagogical_method_english="Extraction Failed",
                confidence=0.0,
                reasoning=f"JSON 파싱 실패: {str(e)}",
            )

    # =========================================================================
    # 카테고리 분류
    # =========================================================================

    def _classify_categories(
        self,
        keywords: Set[str],
        abstract: str,
    ) -> List[str]:
        """키워드와 초록 기반 카테고리 분류"""
        categories = set()

        # 키워드 매칭
        all_text = " ".join(keywords) + " " + abstract.lower()

        # 교육 수준 분류
        for keyword, level in KEYWORD_TO_LEVEL.items():
            if keyword in all_text:
                categories.add(level.value)

        # 분야 분류
        for keyword, domain in KEYWORD_TO_DOMAIN.items():
            if keyword in all_text:
                categories.add(domain.value)

        return list(categories)

    # =========================================================================
    # 검증 및 Seed 생성
    # =========================================================================

    def _create_seed(
        self,
        llm_result: LLMExtractionResult,
        categories: List[str],
        source_record_no: int,
    ) -> ScenarioSeed:
        """검증을 거쳐 ScenarioSeed 생성"""
        warnings = []
        status = ExtractionStatus.SUCCESS

        # 1. 길이 검증
        if len(llm_result.topic) < self.MIN_TOPIC_LENGTH:
            warnings.append(f"주제가 너무 짧습니다: '{llm_result.topic}'")
        if len(llm_result.topic) > self.MAX_TOPIC_LENGTH:
            warnings.append(f"주제가 너무 깁니다 ({len(llm_result.topic)}자)")

        if len(llm_result.pedagogical_method) < self.MIN_METHOD_LENGTH:
            warnings.append(f"교수법이 너무 짧습니다: '{llm_result.pedagogical_method}'")
        if len(llm_result.pedagogical_method) > self.MAX_METHOD_LENGTH:
            warnings.append(f"교수법이 너무 깁니다 ({len(llm_result.pedagogical_method)}자)")

        # 2. 신뢰도 검증
        if llm_result.confidence < self.MIN_CONFIDENCE:
            warnings.append(f"낮은 신뢰도: {llm_result.confidence:.2f}")

        # 3. 모호성 검증
        vague_terms = ["general", "various", "multiple", "다양한", "일반적인", "여러"]
        for term in vague_terms:
            if term in llm_result.topic.lower():
                warnings.append(f"주제에 모호한 표현 포함: '{term}'")
            if term in llm_result.pedagogical_method.lower():
                warnings.append(f"교수법에 모호한 표현 포함: '{term}'")

        # 4. 추출 실패 체크
        if "추출 실패" in llm_result.topic or "Extraction Failed" in llm_result.topic_english:
            status = ExtractionStatus.FAILED
            warnings.append("LLM 추출 실패")
        elif warnings:
            status = ExtractionStatus.NEEDS_REVIEW

        return ScenarioSeed(
            topic=llm_result.topic,
            pedagogical_method=llm_result.pedagogical_method,
            categories=categories,
            source_record_no=source_record_no,
            status=status,
            warnings=warnings,
            confidence_score=llm_result.confidence,
        )

    # =========================================================================
    # Seed DB 저장/로드
    # =========================================================================

    @staticmethod
    def save_seeds(seeds: List[ScenarioSeed], path: str) -> None:
        """시드 목록을 JSON 파일로 저장"""
        data = {
            "created_at": datetime.now().isoformat(),
            "total_count": len(seeds),
            "success_count": len([s for s in seeds if s.status == ExtractionStatus.SUCCESS]),
            "review_count": len([s for s in seeds if s.status == ExtractionStatus.NEEDS_REVIEW]),
            "failed_count": len([s for s in seeds if s.status == ExtractionStatus.FAILED]),
            "seeds": [s.to_dict() for s in seeds],
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_seeds(path: str) -> List[ScenarioSeed]:
        """JSON 파일에서 시드 목록 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return [ScenarioSeed.from_dict(s) for s in data["seeds"]]

    # =========================================================================
    # 통계
    # =========================================================================

    @staticmethod
    def get_stats(seeds: List[ScenarioSeed]) -> Dict:
        """시드 통계 정보"""
        return {
            "total": len(seeds),
            "success": len([s for s in seeds if s.status == ExtractionStatus.SUCCESS]),
            "needs_review": len([s for s in seeds if s.status == ExtractionStatus.NEEDS_REVIEW]),
            "failed": len([s for s in seeds if s.status == ExtractionStatus.FAILED]),
            "avg_confidence": (
                sum(s.confidence_score for s in seeds) / len(seeds)
                if seeds else 0.0
            ),
            "categories": list(set(
                cat for s in seeds for cat in s.categories
            )),
        }
