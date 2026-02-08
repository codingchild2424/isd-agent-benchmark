"""
ADDIE 기반 평가 데이터 모델
"""

from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, computed_field


class ADDIEPhase(str, Enum):
    """ADDIE 단계"""
    ANALYSIS = "analysis"
    DESIGN = "design"
    DEVELOPMENT = "development"
    IMPLEMENTATION = "implementation"
    EVALUATION = "evaluation"


class ScoreLevel(str, Enum):
    """평가 등급 (0-10점 척도)"""
    EXCELLENT = "excellent"        # 9-10점: 매우우수
    GOOD = "good"                  # 7-8점: 우수
    SATISFACTORY = "satisfactory"  # 5-6점: 보통
    POOR = "poor"                  # 3-4점: 미흡
    ABSENT = "absent"              # 1-2점: 부재


def get_score_level(score: float) -> ScoreLevel:
    """점수에서 등급 계산"""
    if score >= 9:
        return ScoreLevel.EXCELLENT
    elif score >= 7:
        return ScoreLevel.GOOD
    elif score >= 5:
        return ScoreLevel.SATISFACTORY
    elif score >= 3:
        return ScoreLevel.POOR
    else:
        return ScoreLevel.ABSENT


class RubricItem(BaseModel):
    """개별 평가 항목"""
    item_id: str = Field(..., description="항목 ID (예: A1, D2)")
    phase: ADDIEPhase = Field(..., description="ADDIE 단계")
    name: str = Field(..., description="항목명")
    description: str = Field("", description="평가 기준 설명")
    score: float = Field(0.0, ge=0, le=10, description="점수 (0-10)")
    reasoning: Optional[str] = Field(None, description="평가 근거")

    @computed_field
    @property
    def level(self) -> ScoreLevel:
        """점수에서 등급 자동 계산"""
        return get_score_level(self.score)


class PhaseScore(BaseModel):
    """ADDIE 단계별 점수"""
    phase: ADDIEPhase
    items: List[RubricItem]
    raw_score: float = Field(..., description="원점수 합계")
    weighted_score: float = Field(..., description="가중치 적용 점수")
    max_score: float = Field(..., description="최대 가능 점수")

    @computed_field
    @property
    def percentage(self) -> float:
        """백분율 점수"""
        return (self.raw_score / self.max_score) * 100 if self.max_score > 0 else 0

    @computed_field
    @property
    def average_score(self) -> float:
        """평균 점수"""
        return self.raw_score / len(self.items) if self.items else 0


class ADDIEScore(BaseModel):
    """ADDIE 종합 평가 점수"""
    analysis: PhaseScore
    design: PhaseScore
    development: PhaseScore
    implementation: PhaseScore
    evaluation: PhaseScore

    total_raw: float = Field(..., description="원점수 총합")
    total_weighted: float = Field(..., description="가중치 적용 총점")
    normalized_score: float = Field(..., description="정규화 점수 (0-100)")

    # 평가 메타데이터
    strengths: List[str] = Field(default_factory=list, description="강점")
    improvements: List[str] = Field(default_factory=list, description="개선점")
    overall_assessment: str = Field("", description="종합 평가")

    @property
    def phases(self) -> Dict[ADDIEPhase, PhaseScore]:
        return {
            ADDIEPhase.ANALYSIS: self.analysis,
            ADDIEPhase.DESIGN: self.design,
            ADDIEPhase.DEVELOPMENT: self.development,
            ADDIEPhase.IMPLEMENTATION: self.implementation,
            ADDIEPhase.EVALUATION: self.evaluation,
        }

    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return {
            "phases": {
                phase.value: {
                    "items": [item.model_dump() for item in ps.items],
                    "raw_score": ps.raw_score,
                    "weighted_score": ps.weighted_score,
                    "percentage": ps.percentage,
                    "average_score": ps.average_score,
                }
                for phase, ps in self.phases.items()
            },
            "total_raw": self.total_raw,
            "total_weighted": self.total_weighted,
            "normalized_score": self.normalized_score,
            "strengths": self.strengths,
            "improvements": self.improvements,
            "overall_assessment": self.overall_assessment,
        }


class ContextProfile(BaseModel):
    """시나리오 컨텍스트 프로필"""
    learner_age: Optional[str] = None
    learner_education: Optional[str] = None
    domain_expertise: Optional[str] = None
    # 학습자 직업/역할 (Context Matrix 13-16번)
    learner_role: Optional[str] = None
    institution_type: Optional[str] = None
    delivery_mode: Optional[str] = None
    class_size: Optional[str] = None
    evaluation_focus: Optional[str] = None
    tech_environment: Optional[str] = None
    duration: Optional[str] = None
    # 교육 도메인 (Context Matrix 23-32번)
    education_domain: Optional[str] = None


class TrajectoryScore(BaseModel):
    """궤적 평가 점수 (BFCL 기반)"""
    tool_correctness: float = Field(0.0, ge=0, le=25, description="도구 정확성")
    argument_accuracy: float = Field(0.0, ge=0, le=25, description="인자 정확성")
    redundancy_avoidance: float = Field(0.0, ge=0, le=25, description="중복 회피")
    result_utilization: float = Field(0.0, ge=0, le=25, description="결과 활용도")

    @computed_field
    @property
    def total(self) -> float:
        """총점 (0-100)"""
        return (
            self.tool_correctness +
            self.argument_accuracy +
            self.redundancy_avoidance +
            self.result_utilization
        )

    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return {
            "tool_correctness": self.tool_correctness,
            "argument_accuracy": self.argument_accuracy,
            "redundancy_avoidance": self.redundancy_avoidance,
            "result_utilization": self.result_utilization,
            "total": self.total,
        }


class CompositeScore(BaseModel):
    """복합 평가 점수"""
    addie: ADDIEScore = Field(..., description="ADDIE 루브릭 평가")
    trajectory: Optional[TrajectoryScore] = Field(None, description="궤적 평가")

    addie_weight: float = Field(default=0.7, description="ADDIE 가중치")
    trajectory_weight: float = Field(default=0.3, description="궤적 가중치")

    context_profile: Optional[ContextProfile] = Field(None, description="컨텍스트")

    @computed_field
    @property
    def total(self) -> float:
        """가중 총점 (0-100)"""
        if self.trajectory:
            return (
                self.addie.normalized_score * self.addie_weight +
                self.trajectory.total * self.trajectory_weight
            )
        return self.addie.normalized_score

    @computed_field
    @property
    def addie_score(self) -> float:
        """ADDIE 점수"""
        return self.addie.normalized_score

    @computed_field
    @property
    def trajectory_score(self) -> Optional[float]:
        """궤적 점수"""
        return self.trajectory.total if self.trajectory else None


# 기존 호환성을 위한 별칭
CompositeScoreV2 = CompositeScore
