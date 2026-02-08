"""
Skill-Tree 기반 학습자 역량 모델링

EduPlanner 논문의 Skill-Tree 구조를 교수설계 맥락으로 변환하여 적용합니다.

원본 (수학 능력):
- Numerical Calculation, Abstract Thinking, Logical Reasoning,
  Analogy Association, Spatial Imagination

변환 (학습자 역량):
- 사전 지식 수준, 학습 선호도, 동기 수준, 자기주도성, 기술 활용 능력
"""

from typing import Optional
from pydantic import BaseModel, Field


class SkillNode(BaseModel):
    """Skill-Tree의 개별 노드"""
    name: str = Field(..., description="역량 이름")
    level: int = Field(..., ge=1, le=5, description="역량 수준 (1-5)")
    description: str = Field(..., description="역량 설명")
    indicators: list[str] = Field(default_factory=list, description="수준별 지표")


class SkillTree(BaseModel):
    """학습자 역량 Skill-Tree"""

    prior_knowledge: SkillNode = Field(
        ...,
        description="사전 지식 수준"
    )
    learning_preference: SkillNode = Field(
        ...,
        description="학습 선호도 (시각/청각/운동감각 등)"
    )
    motivation: SkillNode = Field(
        ...,
        description="학습 동기 수준"
    )
    self_directedness: SkillNode = Field(
        ...,
        description="자기주도 학습 능력"
    )
    tech_literacy: SkillNode = Field(
        ...,
        description="기술/디지털 활용 능력"
    )

    def get_levels(self) -> list[int]:
        """모든 역량 수준을 리스트로 반환"""
        return [
            self.prior_knowledge.level,
            self.learning_preference.level,
            self.motivation.level,
            self.self_directedness.level,
            self.tech_literacy.level,
        ]

    def average_level(self) -> float:
        """평균 역량 수준 계산"""
        levels = self.get_levels()
        return sum(levels) / len(levels)

    def to_prompt_context(self) -> str:
        """프롬프트에 포함할 학습자 프로필 문자열 생성"""
        return f"""## 학습자 역량 프로필 (Skill-Tree)

1. **사전 지식 수준**: {self.prior_knowledge.level}/5
   - {self.prior_knowledge.description}

2. **학습 선호도**: {self.learning_preference.level}/5
   - {self.learning_preference.description}

3. **학습 동기**: {self.motivation.level}/5
   - {self.motivation.description}

4. **자기주도성**: {self.self_directedness.level}/5
   - {self.self_directedness.description}

5. **기술 활용 능력**: {self.tech_literacy.level}/5
   - {self.tech_literacy.description}

**종합 수준**: {self.average_level():.1f}/5
"""


class LearnerProfile(BaseModel):
    """학습자 프로필"""

    profile_id: str = Field(..., description="프로필 ID")
    name: str = Field(..., description="프로필 이름 (예: 신입사원, 초등학생)")
    skill_tree: SkillTree = Field(..., description="역량 Skill-Tree")
    characteristics: list[str] = Field(default_factory=list, description="특성")
    challenges: list[str] = Field(default_factory=list, description="예상 어려움")

    @classmethod
    def from_scenario(
        cls,
        target_audience: str,
        prior_knowledge: Optional[str] = None,
        learning_environment: Optional[str] = None,
    ) -> "LearnerProfile":
        """시나리오 정보로부터 학습자 프로필 생성"""

        # 기본값 설정 (추후 LLM으로 동적 생성 가능)
        skill_tree = cls._infer_skill_tree(target_audience, prior_knowledge)
        characteristics = cls._infer_characteristics(target_audience)
        challenges = cls._infer_challenges(target_audience, prior_knowledge)

        return cls(
            profile_id=f"LP-{hash(target_audience) % 10000:04d}",
            name=target_audience,
            skill_tree=skill_tree,
            characteristics=characteristics,
            challenges=challenges,
        )

    @staticmethod
    def _infer_skill_tree(
        target_audience: str,
        prior_knowledge: Optional[str] = None
    ) -> SkillTree:
        """대상자 정보로부터 Skill-Tree 추론"""

        # 간단한 규칙 기반 추론 (추후 LLM 기반으로 개선 가능)
        audience_lower = target_audience.lower()

        # 기본 수준
        base_levels = {
            "prior_knowledge": 3,
            "learning_preference": 3,
            "motivation": 3,
            "self_directedness": 3,
            "tech_literacy": 3,
        }

        # 대상자별 조정
        if "신입" in audience_lower or "초보" in audience_lower:
            base_levels["prior_knowledge"] = 2
            base_levels["motivation"] = 4
        elif "초등" in audience_lower:
            base_levels["prior_knowledge"] = 2
            base_levels["self_directedness"] = 2
            base_levels["tech_literacy"] = 2
        elif "직장인" in audience_lower or "성인" in audience_lower:
            base_levels["prior_knowledge"] = 3
            base_levels["motivation"] = 4
            base_levels["self_directedness"] = 4
            base_levels["tech_literacy"] = 4
        elif "전문가" in audience_lower or "고급" in audience_lower:
            base_levels["prior_knowledge"] = 5
            base_levels["self_directedness"] = 5

        return SkillTree(
            prior_knowledge=SkillNode(
                name="사전 지식",
                level=base_levels["prior_knowledge"],
                description=prior_knowledge or "해당 분야 기초 지식 보유",
                indicators=[
                    "Lv1: 관련 지식 없음",
                    "Lv2: 기초 개념 이해",
                    "Lv3: 중급 수준",
                    "Lv4: 고급 수준",
                    "Lv5: 전문가 수준",
                ],
            ),
            learning_preference=SkillNode(
                name="학습 선호도",
                level=base_levels["learning_preference"],
                description="다양한 학습 방식에 대한 수용성",
                indicators=[
                    "Lv1: 특정 방식만 선호",
                    "Lv2: 제한적 수용",
                    "Lv3: 보통",
                    "Lv4: 유연한 수용",
                    "Lv5: 모든 방식 적응",
                ],
            ),
            motivation=SkillNode(
                name="학습 동기",
                level=base_levels["motivation"],
                description="학습에 대한 내적/외적 동기 수준",
                indicators=[
                    "Lv1: 동기 부족",
                    "Lv2: 외적 동기 위주",
                    "Lv3: 보통",
                    "Lv4: 높은 동기",
                    "Lv5: 매우 높은 내적 동기",
                ],
            ),
            self_directedness=SkillNode(
                name="자기주도성",
                level=base_levels["self_directedness"],
                description="스스로 학습을 계획하고 실행하는 능력",
                indicators=[
                    "Lv1: 전적으로 지도 필요",
                    "Lv2: 부분적 지도 필요",
                    "Lv3: 보통",
                    "Lv4: 대부분 자기주도",
                    "Lv5: 완전 자기주도",
                ],
            ),
            tech_literacy=SkillNode(
                name="기술 활용",
                level=base_levels["tech_literacy"],
                description="디지털 도구 및 기술 활용 능력",
                indicators=[
                    "Lv1: 기술 활용 어려움",
                    "Lv2: 기본 활용 가능",
                    "Lv3: 보통",
                    "Lv4: 능숙한 활용",
                    "Lv5: 전문적 활용",
                ],
            ),
        )

    @staticmethod
    def _infer_characteristics(target_audience: str) -> list[str]:
        """대상자 특성 추론"""
        audience_lower = target_audience.lower()
        characteristics = []

        if "신입" in audience_lower:
            characteristics.extend([
                "조직 문화 적응 필요",
                "빠른 성장 욕구",
                "실무 적용 중시",
            ])
        elif "초등" in audience_lower:
            characteristics.extend([
                "짧은 집중 시간",
                "게임/놀이 기반 학습 선호",
                "시각적 자료 효과적",
            ])
        elif "직장인" in audience_lower:
            characteristics.extend([
                "시간 제약 있음",
                "실무 적용 중시",
                "효율적 학습 선호",
            ])

        return characteristics

    @staticmethod
    def _infer_challenges(
        target_audience: str,
        prior_knowledge: Optional[str] = None
    ) -> list[str]:
        """예상 어려움 추론"""
        challenges = []
        audience_lower = target_audience.lower()

        if "신입" in audience_lower or "초보" in audience_lower:
            challenges.append("기초 개념 이해 부족 가능")
        if "초등" in audience_lower:
            challenges.append("추상적 개념 이해 어려움")
            challenges.append("장시간 집중 어려움")
        if prior_knowledge and "없" in prior_knowledge:
            challenges.append("선수 학습 필요")

        return challenges


# 사전 정의된 학습자 프로필 템플릿
PROFILE_TEMPLATES = {
    "beginner": LearnerProfile(
        profile_id="TPL-BEG",
        name="초보 학습자",
        skill_tree=SkillTree(
            prior_knowledge=SkillNode(
                name="사전 지식", level=2,
                description="기초 개념만 이해", indicators=[]
            ),
            learning_preference=SkillNode(
                name="학습 선호도", level=3,
                description="시각적 자료 선호", indicators=[]
            ),
            motivation=SkillNode(
                name="학습 동기", level=3,
                description="보통 수준의 동기", indicators=[]
            ),
            self_directedness=SkillNode(
                name="자기주도성", level=2,
                description="지도가 필요함", indicators=[]
            ),
            tech_literacy=SkillNode(
                name="기술 활용", level=3,
                description="기본 활용 가능", indicators=[]
            ),
        ),
        characteristics=["기초부터 시작 필요", "단계별 안내 필요"],
        challenges=["복잡한 개념 이해 어려움"],
    ),
    "intermediate": LearnerProfile(
        profile_id="TPL-INT",
        name="중급 학습자",
        skill_tree=SkillTree(
            prior_knowledge=SkillNode(
                name="사전 지식", level=3,
                description="중급 수준 지식 보유", indicators=[]
            ),
            learning_preference=SkillNode(
                name="학습 선호도", level=4,
                description="다양한 방식 수용", indicators=[]
            ),
            motivation=SkillNode(
                name="학습 동기", level=4,
                description="높은 학습 의지", indicators=[]
            ),
            self_directedness=SkillNode(
                name="자기주도성", level=4,
                description="대부분 자기주도", indicators=[]
            ),
            tech_literacy=SkillNode(
                name="기술 활용", level=4,
                description="능숙한 활용", indicators=[]
            ),
        ),
        characteristics=["심화 학습 가능", "자기주도적"],
        challenges=["고급 개념으로의 도약"],
    ),
    "advanced": LearnerProfile(
        profile_id="TPL-ADV",
        name="고급 학습자",
        skill_tree=SkillTree(
            prior_knowledge=SkillNode(
                name="사전 지식", level=5,
                description="전문가 수준", indicators=[]
            ),
            learning_preference=SkillNode(
                name="학습 선호도", level=5,
                description="모든 방식 적응", indicators=[]
            ),
            motivation=SkillNode(
                name="학습 동기", level=5,
                description="매우 높은 내적 동기", indicators=[]
            ),
            self_directedness=SkillNode(
                name="자기주도성", level=5,
                description="완전 자기주도", indicators=[]
            ),
            tech_literacy=SkillNode(
                name="기술 활용", level=5,
                description="전문적 활용", indicators=[]
            ),
        ),
        characteristics=["전문성 심화", "리더십 역할 가능"],
        challenges=["새로운 도전 필요"],
    ),
}
