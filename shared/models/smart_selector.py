"""
지능형 선택기 (Smart Selector)

동일한 IDLD 시드(Seed)에서 다양한 컨텍스트 변형을 생성하여
에이전트의 맥락 적응력을 테스트합니다.

예: "머신러닝 기초 교육" 시드 →
    - 초등학생 + 초등학교 + 오프라인
    - 대학생 + 대학교 + 블렌디드
    - 직장인 + 기업 + 온라인 비실시간
    - 연구원 + 연구소 + 자기주도

variant_type 메타데이터를 포함한 컨텍스트 변형 생성 지원
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .context_matrix import ContextMatrix, ContextCombination, SUB_DIMENSION_TO_FIELD
from .context_filter import ContextFilter
from .seed_extractor import ScenarioSeed


# =============================================================================
# 변형 유형 정의
# =============================================================================

class VariantType(str, Enum):
    """
    시나리오 변형 유형

    - IDLD_ALIGNED: IDLD 논문의 맥락과 일치하는 전형적인 조합
    - CONTEXT_VARIANT: 컨텍스트를 의도적으로 변형한 확장 케이스
    """
    IDLD_ALIGNED = "idld_aligned"
    CONTEXT_VARIANT = "context_variant"


@dataclass
class TaggedContext:
    """variant_type 태그가 붙은 컨텍스트 조합"""
    context: ContextCombination
    variant_type: VariantType
    generation_source: str = ""  # 생성 전략 출처 (예: "institution_learner_pair", "challenging")

    def to_dict(self) -> Dict:
        return {
            "context": self.context.to_dict(),
            "variant_type": self.variant_type.value,
            "generation_source": self.generation_source,
        }


# =============================================================================
# 변형 전략 정의
# =============================================================================

# 현실적인 기관-학습자 매칭 쌍
INSTITUTION_LEARNER_PAIRS: List[Tuple[str, str, str]] = [
    # (기관유형, 연령, 역할)
    ("초·중등학교", "10대", "학생/취준생"),
    ("대학교(학부)", "20대", "학생/취준생"),
    ("대학원", "20대", "학생/취준생"),
    ("기업", "30대", "현직 실무자"),
    ("기업", "40대 이상", "관리자/리더"),
    ("직업훈련기관", "30대", "현직 실무자"),
    ("직업훈련기관", "20대", "학생/취준생"),
    ("공공/비영리 교육기관", "40대 이상", "일반 시민"),
    ("공공/비영리 교육기관", "30대", "현직 실무자"),
]

# 전달방식-규모-기간 조합
DELIVERY_COMBINATIONS: List[Tuple[str, str, str]] = [
    # (전달방식, 규모, 기간)
    ("오프라인(교실 수업)", "중규모(10–30명)", "중기 과정(2–4주)"),
    ("온라인 비실시간(LMS)", "대규모(30명 이상)", "장기 과정(1~6개월)"),
    ("온라인 실시간(Zoom 등)", "소규모(1–10명)", "단기 집중 과정(1주 내)"),
    ("블렌디드(혼합형)", "중규모(10–30명)", "중기 과정(2–4주)"),
    ("모바일 마이크로러닝", "대규모(30명 이상)", "단기 집중 과정(1주 내)"),
    ("프로젝트 기반(PBL)", "소규모(1–10명)", "장기 과정(1~6개월)"),
    ("시뮬레이션/VR 기반", "소규모(1–10명)", "중기 과정(2–4주)"),
    ("자기주도 학습", "소규모(1–10명)", "유동적 일정"),
]

# 도전적 조합 (일반적이지 않지만 가능한 조합)
CHALLENGING_COMBINATIONS: List[Dict[str, str]] = [
    # 어린 학습자 + 고기술 환경
    {
        "learner_age": "10대",
        "tech_environment": "풀 기술 지원(LMS, 스마트기기, 인터넷)",
        "delivery_mode": "시뮬레이션/VR 기반",
    },
    # 성인 + 제한적 기술 환경
    {
        "learner_age": "40대 이상",
        "tech_environment": "제한적 기술 환경(PC 미보유, 스마트폰 위주)",
        "delivery_mode": "모바일 마이크로러닝",
    },
    # 대규모 + 프로젝트 기반
    {
        "class_size": "대규모(30명 이상)",
        "delivery_mode": "프로젝트 기반(PBL)",
        "duration": "장기 과정(1~6개월)",
    },
    # 소규모 + 온라인 비실시간
    {
        "class_size": "소규모(1–10명)",
        "delivery_mode": "온라인 비실시간(LMS)",
        "duration": "자기주도 일정",
    },
]


# =============================================================================
# SmartSelector 클래스
# =============================================================================

@dataclass
class VariantResult:
    """변형 생성 결과"""
    tagged_variants: List[TaggedContext]  # variant_type 포함
    seed: ScenarioSeed
    total_generated: int = 0
    filtered_count: int = 0
    duplicate_count: int = 0

    @property
    def variants(self) -> List[ContextCombination]:
        """하위 호환성: ContextCombination 리스트 반환"""
        return [tv.context for tv in self.tagged_variants]

    @property
    def success_count(self) -> int:
        return len(self.tagged_variants)

    @property
    def idld_aligned_count(self) -> int:
        """idld_aligned 변형 수"""
        return sum(1 for tv in self.tagged_variants if tv.variant_type == VariantType.IDLD_ALIGNED)

    @property
    def context_variant_count(self) -> int:
        """context_variant 변형 수"""
        return sum(1 for tv in self.tagged_variants if tv.variant_type == VariantType.CONTEXT_VARIANT)

    def get_by_type(self, variant_type: VariantType) -> List[TaggedContext]:
        """특정 유형의 변형만 반환"""
        return [tv for tv in self.tagged_variants if tv.variant_type == variant_type]

    def summary(self) -> Dict:
        return {
            "seed_topic": self.seed.topic,
            "total_generated": self.total_generated,
            "filtered_by_rules": self.filtered_count,
            "duplicates_removed": self.duplicate_count,
            "final_variants": self.success_count,
            "idld_aligned_count": self.idld_aligned_count,
            "context_variant_count": self.context_variant_count,
        }


class SmartSelector:
    """
    지능형 선택기

    동일한 시드에서 다양한 컨텍스트 변형을 생성합니다.
    ContextFilter를 사용하여 비현실적 조합을 자동 배제합니다.

    사용 예:
        selector = SmartSelector()

        # 단일 시드에서 변형 생성
        result = selector.generate_variants(seed, n=10)
        for variant in result.variants:
            print(variant.to_dict())

        # 여러 시드에서 배치 생성
        results = selector.generate_batch(seeds, variants_per_seed=5)
    """

    def __init__(
        self,
        context_matrix: Optional[ContextMatrix] = None,
        context_filter: Optional[ContextFilter] = None,
    ):
        """
        Args:
            context_matrix: 컨텍스트 매트릭스 (None이면 기본 생성)
            context_filter: 컨텍스트 필터 (None이면 기본 생성)
        """
        self.context_matrix = context_matrix or ContextMatrix()
        self.context_filter = context_filter or ContextFilter()

    # =========================================================================
    # 메인 API
    # =========================================================================

    def generate_variants(
        self,
        seed: ScenarioSeed,
        n: int = 10,
        include_challenging: bool = True,
    ) -> VariantResult:
        """
        단일 시드에서 다양한 컨텍스트 변형 생성

        Args:
            seed: 시나리오 시드
            n: 생성할 변형 수
            include_challenging: 도전적 조합 포함 여부

        Returns:
            VariantResult (TaggedContext 리스트 및 통계)
        """
        all_candidates: List[TaggedContext] = []
        used_keys: Set[str] = set()

        # 1. IDLD 정렬 조합 생성 (기관-학습자 매칭 기반)
        aligned = self._generate_realistic_variants()
        for combo in aligned:
            all_candidates.append(TaggedContext(
                context=combo,
                variant_type=VariantType.IDLD_ALIGNED,
                generation_source="institution_learner_pair",
            ))

        # 2. 전달방식 조합 생성 (idld_aligned)
        delivery_based = self._generate_delivery_variants()
        for combo in delivery_based:
            all_candidates.append(TaggedContext(
                context=combo,
                variant_type=VariantType.IDLD_ALIGNED,
                generation_source="delivery_combination",
            ))

        # 3. 컨텍스트 변형 조합 추가
        if include_challenging:
            variants = self._generate_challenging_variants()
            for combo in variants:
                all_candidates.append(TaggedContext(
                    context=combo,
                    variant_type=VariantType.CONTEXT_VARIANT,
                    generation_source="context_variant_combination",
                ))

        # 4. 추가 랜덤 조합 (목표 수의 2배까지) - idld_aligned로 태깅
        while len(all_candidates) < n * 2:
            combo = self.context_matrix.sample_combination()
            all_candidates.append(TaggedContext(
                context=combo,
                variant_type=VariantType.IDLD_ALIGNED,
                generation_source="random_fill",
            ))

        total_generated = len(all_candidates)

        # 5. ContextFilter로 비현실적 조합 필터링
        compatible: List[TaggedContext] = []
        for tagged in all_candidates:
            result = self.context_filter.check_compatibility(seed, tagged.context)
            if result.is_compatible:
                compatible.append(tagged)

        filtered_count = total_generated - len(compatible)

        # 6. 중복 제거
        unique_variants: List[TaggedContext] = []
        for tagged in compatible:
            key = self._combination_key(tagged.context)
            if key not in used_keys:
                unique_variants.append(tagged)
                used_keys.add(key)

        duplicate_count = len(compatible) - len(unique_variants)

        # 7. 다양성을 위해 셔플 후 n개 선택
        random.shuffle(unique_variants)
        selected = unique_variants[:n]

        return VariantResult(
            tagged_variants=selected,
            seed=seed,
            total_generated=total_generated,
            filtered_count=filtered_count,
            duplicate_count=duplicate_count,
        )

    def generate_batch(
        self,
        seeds: List[ScenarioSeed],
        variants_per_seed: int = 5,
        include_challenging: bool = True,
    ) -> List[VariantResult]:
        """
        여러 시드에서 배치 변형 생성

        Args:
            seeds: 시나리오 시드 리스트
            variants_per_seed: 시드당 생성할 변형 수
            include_challenging: 도전적 조합 포함 여부

        Returns:
            VariantResult 리스트
        """
        results = []
        for seed in seeds:
            result = self.generate_variants(
                seed,
                n=variants_per_seed,
                include_challenging=include_challenging,
            )
            results.append(result)
        return results

    def generate_variant_set(
        self,
        seed: ScenarioSeed,
        target_contexts: List[Dict[str, str]],
    ) -> List[ContextCombination]:
        """
        지정된 컨텍스트 조합으로 변형 생성

        이슈의 V1~V4 예시처럼 특정 조합을 지정하여 생성

        Args:
            seed: 시나리오 시드
            target_contexts: 생성할 컨텍스트 조합 리스트
                예: [
                    {"learner_age": "10대", "institution_type": "초등학교"},
                    {"learner_age": "20대", "institution_type": "대학교"},
                ]

        Returns:
            ContextCombination 리스트 (필터링 통과한 것만)
        """
        variants = []

        for target in target_contexts:
            # 기본 조합 생성
            combo = self.context_matrix.sample_combination()

            # 지정된 필드 덮어쓰기
            for field_name, value in target.items():
                combo.set_field(field_name, value)

            # 필터 통과 확인
            result = self.context_filter.check_compatibility(seed, combo)
            if result.is_compatible:
                variants.append(combo)

        return variants

    # =========================================================================
    # 변형 생성 전략
    # =========================================================================

    def _generate_realistic_variants(self) -> List[ContextCombination]:
        """현실적인 기관-학습자 매칭 기반 변형 생성"""
        variants = []

        for inst, age, role in INSTITUTION_LEARNER_PAIRS:
            combo = ContextCombination(
                institution_type=inst,
                learner_age=age,
                learner_role=role,
            )
            # 나머지 필드 랜덤 채우기
            self._fill_remaining_fields(combo)
            variants.append(combo)

        return variants

    def _generate_delivery_variants(self) -> List[ContextCombination]:
        """전달방식-규모-기간 기반 변형 생성"""
        variants = []

        for delivery, size, duration in DELIVERY_COMBINATIONS:
            combo = ContextCombination(
                delivery_mode=delivery,
                class_size=size,
                duration=duration,
            )
            self._fill_remaining_fields(combo)
            variants.append(combo)

        return variants

    def _generate_challenging_variants(self) -> List[ContextCombination]:
        """도전적 조합 생성"""
        variants = []

        for challenge_dict in CHALLENGING_COMBINATIONS:
            combo = ContextCombination()
            for field_name, value in challenge_dict.items():
                combo.set_field(field_name, value)
            self._fill_remaining_fields(combo)
            variants.append(combo)

        return variants

    def _fill_remaining_fields(self, combo: ContextCombination) -> None:
        """조합의 빈 필드를 랜덤으로 채우기"""
        for sub_dim, field_name in SUB_DIMENSION_TO_FIELD.items():
            if combo.get_field(field_name) is None:
                # 해당 중단계의 옵션 찾기
                for dimension in self.context_matrix.dimensions.values():
                    if sub_dim in dimension:
                        options = dimension[sub_dim]
                        if options:
                            combo.set_field(field_name, random.choice(options))
                        break

    def _combination_key(self, combo: ContextCombination) -> str:
        """조합의 고유 키 생성 (중복 체크용)"""
        return "|".join(str(v) for v in combo.to_dict().values())

    # =========================================================================
    # 통계 및 디버깅
    # =========================================================================

    def explain_variants(self, result: VariantResult) -> str:
        """변형 결과 설명"""
        lines = [
            f"시드: '{result.seed.topic}'",
            f"카테고리: {result.seed.categories}",
            "",
            f"총 생성: {result.total_generated}",
            f"필터링됨: {result.filtered_count}",
            f"중복 제거: {result.duplicate_count}",
            f"최종 변형: {result.success_count}",
            f"  - idld_aligned: {result.idld_aligned_count}",
            f"  - context_variant: {result.context_variant_count}",
            "",
            "생성된 변형:",
        ]

        for i, tagged in enumerate(result.tagged_variants, 1):
            d = tagged.context.to_dict()
            type_marker = "🟢" if tagged.variant_type == VariantType.IDLD_ALIGNED else "🔴"
            lines.append(f"  [{i}] {type_marker} {tagged.variant_type.value} | "
                        f"기관={d['institution_type']}, "
                        f"학습자={d['learner_age']}/{d['learner_role']}, "
                        f"전달={d['delivery_mode']}")

        return "\n".join(lines)

    def get_strategy_summary(self) -> Dict:
        """선택 전략 요약"""
        return {
            "institution_learner_pairs": len(INSTITUTION_LEARNER_PAIRS),
            "delivery_combinations": len(DELIVERY_COMBINATIONS),
            "challenging_combinations": len(CHALLENGING_COMBINATIONS),
            "context_matrix_loaded": len(self.context_matrix.items) > 0,
            "filter_rules": self.context_filter.get_constraint_summary(),
        }
