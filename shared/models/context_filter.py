"""
컨텍스트 필터링 및 충돌 방지 로직

IDLD 시드(Seed)와 Context Matrix를 매핑할 때
논리적으로 모순되는 조합을 필터링합니다.

예: "Higher Education" 시드 → "초·중등학교" 컨텍스트 제외
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from .context_matrix import ContextCombination
from .seed_extractor import ScenarioSeed, EducationLevel, SubjectDomain


# =============================================================================
# 충돌 규칙 정의 (Conflict Rules)
# =============================================================================

# 교육 수준별 제외 컨텍스트
# 형식: {카테고리: {필드명: [제외값들]}}
EDUCATION_LEVEL_CONSTRAINTS: Dict[str, Dict[str, List[str]]] = {
    # K-12 콘텐츠는 기업/공공기관 맥락에서 부적절
    "K-12": {
        "institution_type": ["기업", "공공/비영리 교육기관", "대학원"],
        "learner_age": ["30대", "40대 이상"],
        "learner_role": ["현직 실무자", "관리자/리더", "전문 기술직"],
    },

    # 고등교육 콘텐츠는 초중등 맥락에서 부적절
    "Higher Education": {
        "institution_type": ["초·중등학교"],
        "learner_age": ["10대"],  # 대학생은 20대 이상
    },

    # 대학원 수준은 더 제한적
    "Graduate": {
        "institution_type": ["초·중등학교", "직업훈련기관"],
        "learner_age": ["10대"],
        "domain_expertise": ["초급"],
    },

    # 전문가 개발은 학교 맥락 부적절
    "Professional Development": {
        "institution_type": ["초·중등학교"],
        "learner_age": ["10대"],
        "learner_role": ["학생/취준생"],
    },

    # 기업 교육은 학교 맥락 부적절
    "Corporate Training": {
        "institution_type": ["초·중등학교", "대학교(학부)"],
        "learner_age": ["10대"],
        "learner_role": ["학생/취준생", "예비 교사/교사"],
    },
}

# 분야별 제외 컨텍스트
DOMAIN_CONSTRAINTS: Dict[str, Dict[str, List[str]]] = {
    # 의료/간호는 어린 학습자에게 부적절
    "Healthcare": {
        "learner_age": ["10대"],
        "institution_type": ["초·중등학교"],
    },

    # IT/컴퓨터 과학 고급 과정은 초보자 부적절
    "IT/Computer Science": {
        # 고급 IT는 domain_expertise가 중급 이상이어야 함
        # 이건 domain_expertise "초급"을 제외하는 것으로 처리
    },

    # 비즈니스는 초중등에 부적절
    "Business": {
        "institution_type": ["초·중등학교"],
        "learner_age": ["10대"],
    },
}

# 고급 콘텐츠 키워드 (이런 키워드가 있으면 초급자 제외)
ADVANCED_CONTENT_KEYWORDS: Set[str] = {
    "advanced", "expert", "professional", "specialized",
    "고급", "전문가", "심화", "고급자", "숙련",
    "surgery", "surgical", "clinical practice",
    "machine learning", "deep learning", "neural network",
    "enterprise", "architecture",
}

# 초급 콘텐츠 키워드 (이런 키워드가 있으면 고급자 배제 필요 없음)
BEGINNER_CONTENT_KEYWORDS: Set[str] = {
    "introduction", "beginner", "basic", "fundamental", "elementary",
    "입문", "기초", "초급", "시작하기", "첫걸음",
}


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class ContextConstraint:
    """컨텍스트 제약 조건"""
    field_name: str           # 제약이 적용되는 필드 (예: "institution_type")
    excluded_values: List[str]  # 제외할 값들
    reason: str = ""          # 제외 이유 (디버깅용)

    def is_violated(self, context: ContextCombination) -> bool:
        """컨텍스트가 이 제약을 위반하는지 확인"""
        value = context.get_field(self.field_name)
        if value is None:
            return False
        return value in self.excluded_values


@dataclass
class FilterResult:
    """필터링 결과"""
    is_compatible: bool
    violations: List[str] = field(default_factory=list)
    applied_constraints: List[ContextConstraint] = field(default_factory=list)

    def add_violation(self, constraint: ContextConstraint, value: str) -> None:
        """위반 사항 추가"""
        self.violations.append(
            f"{constraint.field_name}='{value}' 제외됨 ({constraint.reason})"
        )
        self.applied_constraints.append(constraint)
        self.is_compatible = False


# =============================================================================
# ContextFilter 클래스
# =============================================================================

class ContextFilter:
    """
    컨텍스트 필터링 클래스

    시드의 카테고리를 기반으로 호환되지 않는 컨텍스트 조합을 필터링합니다.

    사용 예:
        filter = ContextFilter()

        # 단일 검증
        result = filter.check_compatibility(seed, context)
        if not result.is_compatible:
            print(f"위반: {result.violations}")

        # 배치 필터링
        compatible = filter.filter_compatible_contexts(seed, contexts)
    """

    def __init__(
        self,
        education_constraints: Optional[Dict] = None,
        domain_constraints: Optional[Dict] = None,
    ):
        """
        Args:
            education_constraints: 교육 수준별 제약 (기본값 사용 시 None)
            domain_constraints: 분야별 제약 (기본값 사용 시 None)
        """
        self.education_constraints = education_constraints or EDUCATION_LEVEL_CONSTRAINTS
        self.domain_constraints = domain_constraints or DOMAIN_CONSTRAINTS

    # =========================================================================
    # 메인 API
    # =========================================================================

    def check_compatibility(
        self,
        seed: ScenarioSeed,
        context: ContextCombination,
    ) -> FilterResult:
        """
        시드와 컨텍스트의 호환성 검사

        Args:
            seed: 시나리오 시드
            context: 컨텍스트 조합

        Returns:
            FilterResult (호환 여부 및 위반 사항)
        """
        result = FilterResult(is_compatible=True)

        # 1. 교육 수준 제약 검사
        for category in seed.categories:
            if category in self.education_constraints:
                constraints = self._build_constraints(
                    self.education_constraints[category],
                    reason=f"교육수준 '{category}' 제약"
                )
                self._apply_constraints(context, constraints, result)

        # 2. 분야 제약 검사
        for category in seed.categories:
            if category in self.domain_constraints:
                constraints = self._build_constraints(
                    self.domain_constraints[category],
                    reason=f"분야 '{category}' 제약"
                )
                self._apply_constraints(context, constraints, result)

        # 3. 고급 콘텐츠 검사
        if self._is_advanced_content(seed):
            constraints = [
                ContextConstraint(
                    field_name="domain_expertise",
                    excluded_values=["초급"],
                    reason="고급 콘텐츠는 초급자 제외"
                )
            ]
            self._apply_constraints(context, constraints, result)

        return result

    def filter_compatible_contexts(
        self,
        seed: ScenarioSeed,
        contexts: List[ContextCombination],
    ) -> List[ContextCombination]:
        """
        호환되는 컨텍스트만 필터링

        Args:
            seed: 시나리오 시드
            contexts: 컨텍스트 조합 리스트

        Returns:
            호환되는 컨텍스트 리스트
        """
        compatible = []
        for context in contexts:
            result = self.check_compatibility(seed, context)
            if result.is_compatible:
                compatible.append(context)

        return compatible

    def filter_with_details(
        self,
        seed: ScenarioSeed,
        contexts: List[ContextCombination],
    ) -> Tuple[List[ContextCombination], List[Tuple[ContextCombination, FilterResult]]]:
        """
        호환/비호환 컨텍스트를 분리하여 반환

        Args:
            seed: 시나리오 시드
            contexts: 컨텍스트 조합 리스트

        Returns:
            (호환 리스트, [(비호환 컨텍스트, 위반 결과)])
        """
        compatible = []
        incompatible = []

        for context in contexts:
            result = self.check_compatibility(seed, context)
            if result.is_compatible:
                compatible.append(context)
            else:
                incompatible.append((context, result))

        return compatible, incompatible

    # =========================================================================
    # 제약 조건 빌더
    # =========================================================================

    def get_constraints_for_seed(
        self,
        seed: ScenarioSeed,
    ) -> List[ContextConstraint]:
        """시드에 적용되는 모든 제약 조건 반환"""
        constraints = []

        # 교육 수준 제약
        for category in seed.categories:
            if category in self.education_constraints:
                constraints.extend(
                    self._build_constraints(
                        self.education_constraints[category],
                        reason=f"교육수준 '{category}' 제약"
                    )
                )

        # 분야 제약
        for category in seed.categories:
            if category in self.domain_constraints:
                constraints.extend(
                    self._build_constraints(
                        self.domain_constraints[category],
                        reason=f"분야 '{category}' 제약"
                    )
                )

        # 고급 콘텐츠 제약
        if self._is_advanced_content(seed):
            constraints.append(
                ContextConstraint(
                    field_name="domain_expertise",
                    excluded_values=["초급"],
                    reason="고급 콘텐츠는 초급자 제외"
                )
            )

        return constraints

    def _build_constraints(
        self,
        constraint_dict: Dict[str, List[str]],
        reason: str,
    ) -> List[ContextConstraint]:
        """딕셔너리에서 ContextConstraint 리스트 생성"""
        constraints = []
        for field_name, excluded_values in constraint_dict.items():
            constraints.append(
                ContextConstraint(
                    field_name=field_name,
                    excluded_values=excluded_values,
                    reason=reason,
                )
            )
        return constraints

    def _apply_constraints(
        self,
        context: ContextCombination,
        constraints: List[ContextConstraint],
        result: FilterResult,
    ) -> None:
        """제약 조건 적용 및 결과 업데이트"""
        for constraint in constraints:
            if constraint.is_violated(context):
                value = context.get_field(constraint.field_name)
                result.add_violation(constraint, value)

    # =========================================================================
    # 콘텐츠 분석
    # =========================================================================

    def _is_advanced_content(self, seed: ScenarioSeed) -> bool:
        """시드가 고급 콘텐츠인지 판단"""
        # 주제나 교수법에 고급 키워드 포함 여부
        text = f"{seed.topic} {seed.pedagogical_method}".lower()

        # 초급 키워드가 있으면 고급이 아님
        for keyword in BEGINNER_CONTENT_KEYWORDS:
            if keyword in text:
                return False

        # 고급 키워드가 있으면 고급
        for keyword in ADVANCED_CONTENT_KEYWORDS:
            if keyword in text:
                return True

        return False

    # =========================================================================
    # 규칙 관리
    # =========================================================================

    def add_education_constraint(
        self,
        category: str,
        field_name: str,
        excluded_values: List[str],
    ) -> None:
        """교육 수준 제약 추가"""
        if category not in self.education_constraints:
            self.education_constraints[category] = {}

        if field_name not in self.education_constraints[category]:
            self.education_constraints[category][field_name] = []

        self.education_constraints[category][field_name].extend(excluded_values)

    def add_domain_constraint(
        self,
        category: str,
        field_name: str,
        excluded_values: List[str],
    ) -> None:
        """분야 제약 추가"""
        if category not in self.domain_constraints:
            self.domain_constraints[category] = {}

        if field_name not in self.domain_constraints[category]:
            self.domain_constraints[category][field_name] = []

        self.domain_constraints[category][field_name].extend(excluded_values)

    # =========================================================================
    # 통계 및 디버깅
    # =========================================================================

    def get_constraint_summary(self) -> Dict:
        """제약 조건 요약"""
        education_count = sum(
            len(fields) for fields in self.education_constraints.values()
        )
        domain_count = sum(
            len(fields) for fields in self.domain_constraints.values()
        )

        return {
            "education_categories": list(self.education_constraints.keys()),
            "domain_categories": list(self.domain_constraints.keys()),
            "total_education_rules": education_count,
            "total_domain_rules": domain_count,
            "advanced_keywords": len(ADVANCED_CONTENT_KEYWORDS),
            "beginner_keywords": len(BEGINNER_CONTENT_KEYWORDS),
        }

    def explain_constraints(self, seed: ScenarioSeed) -> str:
        """시드에 적용되는 제약 설명"""
        constraints = self.get_constraints_for_seed(seed)

        if not constraints:
            return f"시드 '{seed.topic}'에 적용되는 제약이 없습니다."

        lines = [f"시드 '{seed.topic}'에 적용되는 제약:"]
        for c in constraints:
            lines.append(f"  - {c.field_name}: {c.excluded_values} 제외 ({c.reason})")

        return "\n".join(lines)
