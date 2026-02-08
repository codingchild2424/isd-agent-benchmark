"""
Context Matrix 데이터 로더 및 조합 생성기

벤치마크용 시나리오 컨텍스트 조합을 생성합니다:
1. Representative Scenarios: 대표적인 교육 상황 조합 (50~100개)
2. Ablation Study: 기준 시나리오에서 한 차원만 변경 (150~250개)
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Set
from itertools import product


@dataclass
class ContextItem:
    """컨텍스트 항목 (소단계)"""
    id: int
    dimension: str      # 대단계: 학습자 특성, 기관 맥락 등
    sub_dimension: str  # 중단계: 연령, 학력수준 등
    value: str          # 소단계: 10대, 20대 등


@dataclass
class ContextCombination:
    """컨텍스트 조합"""
    learner_age: Optional[str] = None           # 연령
    learner_education: Optional[str] = None     # 학력수준
    domain_expertise: Optional[str] = None      # 도메인지식수준
    learner_role: Optional[str] = None          # 직업·역할
    institution_type: Optional[str] = None      # 기관유형
    education_domain: Optional[str] = None      # 교과/직무분야
    delivery_mode: Optional[str] = None         # 전달 방식
    class_size: Optional[str] = None            # 학습자 규모
    evaluation_focus: Optional[str] = None      # 평가 요구
    tech_environment: Optional[str] = None      # 기술 환경
    duration: Optional[str] = None              # 시간·일정

    def to_dict(self) -> Dict[str, Optional[str]]:
        """딕셔너리 변환"""
        return {
            "learner_age": self.learner_age,
            "learner_education": self.learner_education,
            "domain_expertise": self.domain_expertise,
            "learner_role": self.learner_role,
            "institution_type": self.institution_type,
            "education_domain": self.education_domain,
            "delivery_mode": self.delivery_mode,
            "class_size": self.class_size,
            "evaluation_focus": self.evaluation_focus,
            "tech_environment": self.tech_environment,
            "duration": self.duration,
        }

    def copy(self) -> "ContextCombination":
        """복사본 생성"""
        return ContextCombination(**self.to_dict())

    def get_field(self, field_name: str) -> Optional[str]:
        """필드값 조회"""
        return getattr(self, field_name, None)

    def set_field(self, field_name: str, value: str) -> None:
        """필드값 설정"""
        setattr(self, field_name, value)


# 중단계 → 필드명 매핑
SUB_DIMENSION_TO_FIELD = {
    "연령": "learner_age",
    "학력수준": "learner_education",
    "도메인지식수준": "domain_expertise",
    "직업·역할": "learner_role",
    "기관유형": "institution_type",
    "교과분야": "education_domain",
    "직무분야": "education_domain",  # 교과/직무 통합
    "Delivery Mode": "delivery_mode",
    "학습자 규모": "class_size",
    "평가 요구": "evaluation_focus",
    "기술 환경": "tech_environment",
    "시간·일정": "duration",
}


class ContextMatrix:
    """Context Matrix 데이터 로더 및 조합 생성기"""

    def __init__(self, csv_path: Optional[str] = None):
        """
        Args:
            csv_path: Context Matrix CSV 파일 경로. None이면 기본 경로 사용.
        """
        self.items: List[ContextItem] = []
        # 대단계 → 중단계 → 소단계 리스트
        self.dimensions: Dict[str, Dict[str, List[str]]] = {}

        if csv_path:
            self.load_from_csv(csv_path)
        else:
            # 기본 경로 시도
            default_path = self._get_default_csv_path()
            if default_path.exists():
                self.load_from_csv(str(default_path))

    @staticmethod
    def _get_default_csv_path() -> Path:
        """기본 CSV 경로 반환"""
        current_dir = Path(__file__).parent
        # shared/models/ → 3. ISD Agent Benchmark/
        benchmark_dir = current_dir.parent.parent
        return benchmark_dir / "참고문서 " / "Context Matrix - 컨텍스트 축.csv"

    def load_from_csv(self, path: str) -> "ContextMatrix":
        """CSV 파일에서 데이터 로드"""
        self.items = []
        self.dimensions = {}

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = ContextItem(
                    id=int(row['순번']),
                    dimension=row['대단계'],
                    sub_dimension=row['중단계'],
                    value=row['소단계']
                )
                self.items.append(item)

                # 계층 구조 구축
                if item.dimension not in self.dimensions:
                    self.dimensions[item.dimension] = {}
                if item.sub_dimension not in self.dimensions[item.dimension]:
                    self.dimensions[item.dimension][item.sub_dimension] = []
                self.dimensions[item.dimension][item.sub_dimension].append(item.value)

        return self

    def get_dimensions(self) -> List[str]:
        """대단계(차원) 목록 반환"""
        return list(self.dimensions.keys())

    def get_sub_dimensions(self, dimension: str) -> List[str]:
        """중단계 목록 반환"""
        return list(self.dimensions.get(dimension, {}).keys())

    def get_options(self, dimension: str, sub_dimension: str) -> List[str]:
        """소단계(옵션) 목록 반환"""
        return self.dimensions.get(dimension, {}).get(sub_dimension, [])

    def sample_option(self, dimension: str, sub_dimension: str) -> Optional[str]:
        """특정 중단계에서 무작위 옵션 선택"""
        options = self.get_options(dimension, sub_dimension)
        return random.choice(options) if options else None

    def sample_combination(self) -> ContextCombination:
        """모든 중단계에서 무작위로 옵션을 선택하여 조합 생성"""
        combo = ContextCombination()

        for dimension, sub_dims in self.dimensions.items():
            for sub_dim, options in sub_dims.items():
                field_name = SUB_DIMENSION_TO_FIELD.get(sub_dim)
                if field_name and options:
                    # 이미 값이 설정된 경우(교과/직무 통합) 스킵
                    if getattr(combo, field_name, None) is None:
                        setattr(combo, field_name, random.choice(options))

        return combo

    # =========================================================================
    # Representative Scenarios (대표 조합)
    # =========================================================================

    def generate_representative_scenarios(
        self,
        n: int = 100,
        include_edge_cases: bool = True
    ) -> List[ContextCombination]:
        """
        대표적인 교육 상황 조합 생성

        Args:
            n: 생성할 시나리오 수
            include_edge_cases: 극단적인 조합 포함 여부

        Returns:
            ContextCombination 리스트
        """
        scenarios: List[ContextCombination] = []
        used_combinations: Set[str] = set()

        # 1. 현실적인 조합 (Realistic Combinations)
        realistic = self._generate_realistic_combinations()
        for combo in realistic:
            key = self._combination_key(combo)
            if key not in used_combinations:
                scenarios.append(combo)
                used_combinations.add(key)

        # 2. Edge Cases (극단적 조합)
        if include_edge_cases:
            edge_cases = self._generate_edge_cases()
            for combo in edge_cases:
                key = self._combination_key(combo)
                if key not in used_combinations:
                    scenarios.append(combo)
                    used_combinations.add(key)

        # 3. 목표 개수까지 랜덤 조합 추가
        attempts = 0
        max_attempts = n * 10
        while len(scenarios) < n and attempts < max_attempts:
            combo = self.sample_combination()
            key = self._combination_key(combo)
            if key not in used_combinations:
                scenarios.append(combo)
                used_combinations.add(key)
            attempts += 1

        return scenarios[:n]

    def _combination_key(self, combo: ContextCombination) -> str:
        """조합의 고유 키 생성 (중복 체크용)"""
        return "|".join(str(v) for v in combo.to_dict().values())

    def _generate_realistic_combinations(self) -> List[ContextCombination]:
        """현실적인 교육 상황 조합 생성"""
        combinations = []

        # 교육기관 × 연령 매칭
        institution_age_pairs = [
            ("초·중등학교", "10대"),
            ("대학교(학부)", "20대"),
            ("대학원", "20대"),
            ("기업", "30대"),
            ("기업", "40대 이상"),
            ("직업훈련기관", "30대"),
            ("공공/비영리 교육기관", "40대 이상"),
        ]

        for inst, age in institution_age_pairs:
            combo = ContextCombination(
                institution_type=inst,
                learner_age=age,
            )
            # 나머지 필드 랜덤 채우기
            self._fill_remaining_fields(combo)
            combinations.append(combo)

        # 도메인 × 전달방식 매칭
        domain_delivery_pairs = [
            ("AI", "온라인 비실시간(LMS)"),
            ("개발(Software/IT)", "온라인 비실시간(LMS)"),
            ("언어", "오프라인(교실 수업)"),
            ("언어", "온라인 실시간(Zoom 등)"),
            ("의료/간호", "시뮬레이션/VR 기반"),
            ("경영/HR/경영지원", "블렌디드(혼합형)"),
            ("서비스/고객응대", "오프라인(교실 수업)"),
            ("수학", "오프라인(교실 수업)"),
            ("과학", "블렌디드(혼합형)"),
            ("교육(교수·학습)", "블렌디드(혼합형)"),
        ]

        for domain, delivery in domain_delivery_pairs:
            combo = ContextCombination(
                education_domain=domain,
                delivery_mode=delivery,
            )
            self._fill_remaining_fields(combo)
            combinations.append(combo)

        # 규모 × 제약조건 매칭
        size_constraint_pairs = [
            ("소규모(1–10명)", "단기 집중 과정(1주 내)"),
            ("중규모(10–30명)", "중기 과정(2–4주)"),
            ("대규모(30명 이상)", "장기 과정(1~6개월)"),
            ("소규모(1–10명)", "형성평가 중심"),
            ("대규모(30명 이상)", "총괄평가 중심"),
        ]

        for size, constraint in size_constraint_pairs:
            # constraint가 시간인지 평가인지 판단
            if "과정" in constraint:
                combo = ContextCombination(
                    class_size=size,
                    duration=constraint,
                )
            else:
                combo = ContextCombination(
                    class_size=size,
                    evaluation_focus=constraint,
                )
            self._fill_remaining_fields(combo)
            combinations.append(combo)

        return combinations

    def _generate_edge_cases(self) -> List[ContextCombination]:
        """극단적/도전적인 조합 생성"""
        edge_cases = []

        # 연령과 전문성 역설 조합
        edge_cases.append(ContextCombination(
            learner_age="10대",
            domain_expertise="고급",
            delivery_mode="시뮬레이션/VR 기반",
        ))

        edge_cases.append(ContextCombination(
            learner_age="40대 이상",
            domain_expertise="초급",
            tech_environment="제한적 기술 환경(PC 미보유, 스마트폰 위주)",
        ))

        # 규모와 환경 도전 조합
        edge_cases.append(ContextCombination(
            class_size="대규모(30명 이상)",
            delivery_mode="온라인 실시간(Zoom 등)",
            duration="단기 집중 과정(1주 내)",
        ))

        edge_cases.append(ContextCombination(
            class_size="소규모(1–10명)",
            delivery_mode="모바일 마이크로러닝",
            tech_environment="개인 기기 지참(BYOD)",
        ))

        # 전문 분야 특수 조합
        edge_cases.append(ContextCombination(
            education_domain="의료/간호",
            institution_type="직업훈련기관",
            evaluation_focus="프로젝트 기반 평가",
        ))

        edge_cases.append(ContextCombination(
            education_domain="교육(교수·학습)",
            learner_role="예비 교사/교사",
            delivery_mode="프로젝트 기반(PBL)",
        ))

        # 나머지 필드 채우기
        for combo in edge_cases:
            self._fill_remaining_fields(combo)

        return edge_cases

    def _fill_remaining_fields(self, combo: ContextCombination) -> None:
        """조합의 빈 필드를 랜덤으로 채우기"""
        for sub_dim, field_name in SUB_DIMENSION_TO_FIELD.items():
            if getattr(combo, field_name, None) is None:
                # 해당 중단계의 옵션 찾기
                for dimension in self.dimensions.values():
                    if sub_dim in dimension:
                        options = dimension[sub_dim]
                        if options:
                            setattr(combo, field_name, random.choice(options))
                        break

    # =========================================================================
    # Ablation Study (변인 통제 실험)
    # =========================================================================

    def generate_ablation_study(
        self,
        base_scenario: ContextCombination,
        vary_dimension: Optional[str] = None
    ) -> List[ContextCombination]:
        """
        기준 시나리오에서 특정 차원만 변경하는 Ablation 시나리오 생성

        Args:
            base_scenario: 기준 시나리오
            vary_dimension: 변경할 차원 (None이면 모든 차원)

        Returns:
            변형된 ContextCombination 리스트
        """
        variants: List[ContextCombination] = []

        # 변경할 필드 결정
        if vary_dimension:
            target_fields = self._get_fields_for_dimension(vary_dimension)
        else:
            target_fields = list(SUB_DIMENSION_TO_FIELD.values())

        # 중복 제거
        target_fields = list(set(target_fields))

        for field_name in target_fields:
            # 해당 필드의 모든 옵션 가져오기
            options = self._get_options_for_field(field_name)
            current_value = base_scenario.get_field(field_name)

            for option in options:
                if option != current_value:
                    # 기계적 변형: 해당 필드만 변경
                    variant = base_scenario.copy()
                    variant.set_field(field_name, option)
                    variants.append(variant)

        return variants

    def generate_full_ablation_study(
        self,
        base_scenarios: List[ContextCombination]
    ) -> Dict[str, List[ContextCombination]]:
        """
        여러 기준 시나리오에 대해 전체 Ablation Study 생성

        Args:
            base_scenarios: 기준 시나리오 리스트

        Returns:
            {기준시나리오_인덱스: [변형 시나리오들]} 딕셔너리
        """
        result = {}

        for i, base in enumerate(base_scenarios):
            variants = self.generate_ablation_study(base, vary_dimension=None)
            result[f"base_{i}"] = variants

        return result

    def _get_fields_for_dimension(self, dimension: str) -> List[str]:
        """차원에 해당하는 필드명 리스트 반환"""
        fields = []
        if dimension in self.dimensions:
            for sub_dim in self.dimensions[dimension].keys():
                field_name = SUB_DIMENSION_TO_FIELD.get(sub_dim)
                if field_name:
                    fields.append(field_name)
        return fields

    def _get_options_for_field(self, field_name: str) -> List[str]:
        """필드에 해당하는 모든 옵션 반환"""
        # 역매핑: field_name → sub_dimension
        for sub_dim, fname in SUB_DIMENSION_TO_FIELD.items():
            if fname == field_name:
                for dimension in self.dimensions.values():
                    if sub_dim in dimension:
                        return dimension[sub_dim]
        return []

    # =========================================================================
    # 전체 조합 (Iterator)
    # =========================================================================

    def all_combinations(self) -> Iterator[ContextCombination]:
        """
        모든 가능한 조합을 순회하는 Iterator

        주의: 조합 수가 매우 많을 수 있음 (수십만 개)
        """
        # 각 필드별 옵션 수집
        field_options: Dict[str, List[str]] = {}

        for sub_dim, field_name in SUB_DIMENSION_TO_FIELD.items():
            if field_name not in field_options:
                options = self._get_options_for_field(field_name)
                if options:
                    field_options[field_name] = options

        # 모든 조합 생성
        fields = list(field_options.keys())
        option_lists = [field_options[f] for f in fields]

        for combo_values in product(*option_lists):
            combo = ContextCombination()
            for field_name, value in zip(fields, combo_values):
                setattr(combo, field_name, value)
            yield combo

    def count_all_combinations(self) -> int:
        """전체 조합 수 계산"""
        count = 1
        seen_fields = set()

        for sub_dim, field_name in SUB_DIMENSION_TO_FIELD.items():
            if field_name not in seen_fields:
                options = self._get_options_for_field(field_name)
                if options:
                    count *= len(options)
                    seen_fields.add(field_name)

        return count

    # =========================================================================
    # 통계 정보
    # =========================================================================

    def summary(self) -> Dict:
        """데이터 요약 정보 반환"""
        summary = {
            "total_items": len(self.items),
            "dimensions": {},
            "total_combinations": self.count_all_combinations(),
        }

        for dim, sub_dims in self.dimensions.items():
            summary["dimensions"][dim] = {
                sub_dim: len(options)
                for sub_dim, options in sub_dims.items()
            }

        return summary

    def __repr__(self) -> str:
        return f"ContextMatrix(items={len(self.items)}, dimensions={len(self.dimensions)})"
