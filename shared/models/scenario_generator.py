"""
시나리오 생성기

IDLD 데이터셋과 Context Matrix를 연계하여
벤치마크용 교수설계 시나리오를 생성합니다.

Note: 이 모듈은 데이터 준비 및 구조화만 담당.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .idld_dataset import IDLDDataset, IDLDRecord, ScenarioSchema, SourceMapping
from .context_matrix import ContextMatrix, ContextCombination
from .seed_extractor import ScenarioSeed
from .context_filter import ContextFilter


@dataclass
class ScenarioGenerationRequest:
    """시나리오 생성 요청"""
    idld_record: IDLDRecord
    context_combination: ContextCombination
    difficulty: str = "medium"

    def to_llm_prompt_context(self) -> Dict:
        """
        LLM 프롬프트에 전달할 컨텍스트 생성

        주의: 논문 식별 정보(DOI, 저자, 제목 등) 제외
        """
        return {
            # IDLD에서 추출한 교육적 힌트 (익명화)
            "source_abstract": self.idld_record.abstract,
            "source_keywords": list(self.idld_record.keywords),

            # Context Matrix에서 선택된 조합
            "context": self.context_combination.to_dict(),

            # 난이도
            "difficulty": self.difficulty,
        }


class ScenarioGenerator:
    """
    시나리오 생성기

    IDLD 논문 데이터와 Context Matrix를 결합하여
    다양한 교수설계 시나리오를 생성합니다.
    """

    def __init__(
        self,
        idld_dataset: Optional[IDLDDataset] = None,
        context_matrix: Optional[ContextMatrix] = None,
        context_filter: Optional[ContextFilter] = None,
    ):
        self.idld = idld_dataset or IDLDDataset()
        self.context_matrix = context_matrix or ContextMatrix()
        self.context_filter = context_filter or ContextFilter()
        self.source_mapping = SourceMapping()

    # =========================================================================
    # 시나리오 생성 요청 준비
    # =========================================================================

    def prepare_generation_requests(
        self,
        n: int,
        difficulty_distribution: Optional[Dict[str, float]] = None,
        min_abstract_length: int = 200,
    ) -> List[ScenarioGenerationRequest]:
        """
        시나리오 생성 요청 준비

        Args:
            n: 생성할 시나리오 수
            difficulty_distribution: 난이도 분포 (예: {"easy": 0.3, "medium": 0.5, "hard": 0.2})
            min_abstract_length: 최소 초록 길이

        Returns:
            ScenarioGenerationRequest 리스트
        """
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

        # IDLD에서 충분한 초록이 있는 레코드 필터링
        valid_records = self.idld.filter_has_abstract(min_abstract_length)
        sampled_records = random.sample(valid_records, min(n, len(valid_records)))

        # Context Matrix에서 대표 조합 생성
        context_combos = self.context_matrix.generate_representative_scenarios(n=n)

        # 난이도 할당
        difficulties = self._assign_difficulties(n, difficulty_distribution)

        # 요청 생성
        requests = []
        for i, record in enumerate(sampled_records):
            context = context_combos[i % len(context_combos)]
            difficulty = difficulties[i]

            req = ScenarioGenerationRequest(
                idld_record=record,
                context_combination=context,
                difficulty=difficulty,
            )
            requests.append(req)

        return requests

    def prepare_ablation_requests(
        self,
        base_record: IDLDRecord,
        base_context: ContextCombination,
        vary_dimension: str,
    ) -> List[ScenarioGenerationRequest]:
        """
        Ablation Study용 시나리오 생성 요청 준비

        동일한 IDLD 레코드에서, Context Matrix의 한 차원만 변경
        """
        variants = self.context_matrix.generate_ablation_study(
            base_context,
            vary_dimension=vary_dimension
        )

        requests = []
        for context in variants:
            req = ScenarioGenerationRequest(
                idld_record=base_record,
                context_combination=context,
                difficulty="medium",
            )
            requests.append(req)

        return requests

    def prepare_requests_with_seed(
        self,
        seed: ScenarioSeed,
        record: IDLDRecord,
        n_contexts: int = 1,
        difficulty: str = "medium",
    ) -> List[ScenarioGenerationRequest]:
        """
        시드 기반 시나리오 생성 요청 준비 (충돌 필터링 적용)

        Args:
            seed: 시나리오 시드 (카테고리 정보 포함)
            record: IDLD 레코드
            n_contexts: 생성할 컨텍스트 조합 수
            difficulty: 난이도

        Returns:
            ScenarioGenerationRequest 리스트
        """
        # 1. 대표 컨텍스트 조합 생성
        all_contexts = self.context_matrix.generate_representative_scenarios(
            n=n_contexts * 3  # 필터링 후 충분한 수 확보를 위해 여유있게 생성
        )

        # 2. 시드와 호환되는 컨텍스트만 필터링
        compatible_contexts = self.context_filter.filter_compatible_contexts(
            seed, all_contexts
        )

        # 3. 요청 수 만큼 샘플링
        if len(compatible_contexts) < n_contexts:
            # 호환 컨텍스트가 부족하면 있는 것만 사용
            selected_contexts = compatible_contexts
        else:
            selected_contexts = random.sample(compatible_contexts, n_contexts)

        # 4. 요청 생성
        requests = []
        for context in selected_contexts:
            req = ScenarioGenerationRequest(
                idld_record=record,
                context_combination=context,
                difficulty=difficulty,
            )
            requests.append(req)

        return requests

    def prepare_batch_with_seeds(
        self,
        seeds: List[ScenarioSeed],
        records: List[IDLDRecord],
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ) -> List[ScenarioGenerationRequest]:
        """
        여러 시드에 대해 배치 생성 요청 준비

        Args:
            seeds: 시나리오 시드 리스트
            records: IDLD 레코드 리스트 (시드와 1:1 대응)
            difficulty_distribution: 난이도 분포

        Returns:
            ScenarioGenerationRequest 리스트
        """
        if len(seeds) != len(records):
            raise ValueError("시드와 레코드 수가 일치해야 합니다")

        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

        difficulties = self._assign_difficulties(len(seeds), difficulty_distribution)

        requests = []
        for seed, record, difficulty in zip(seeds, records, difficulties):
            reqs = self.prepare_requests_with_seed(
                seed=seed,
                record=record,
                n_contexts=1,
                difficulty=difficulty,
            )
            requests.extend(reqs)

        return requests

    def _assign_difficulties(
        self,
        n: int,
        distribution: Dict[str, float]
    ) -> List[str]:
        """난이도 분포에 따라 할당"""
        difficulties = []
        for diff, ratio in distribution.items():
            count = int(n * ratio)
            difficulties.extend([diff] * count)

        # 부족한 경우 medium으로 채움
        while len(difficulties) < n:
            difficulties.append("medium")

        random.shuffle(difficulties)
        return difficulties[:n]

    # =========================================================================
    # 시나리오 ID 생성
    # =========================================================================

    def generate_scenario_id(
        self,
        prefix: str = "IDLD",
        index: int = 1
    ) -> str:
        """시나리오 ID 생성"""
        return f"{prefix}-{index:04d}"

    # =========================================================================
    # 출처 매핑 관리
    # =========================================================================

    def register_source_mapping(
        self,
        scenario_id: str,
        record: IDLDRecord
    ) -> None:
        """시나리오-출처 매핑 등록"""
        self.source_mapping.add(scenario_id, record)

    def save_source_mapping(self, path: str) -> None:
        """출처 매핑 저장"""
        self.source_mapping.save(path)

    def get_source_for_scenario(self, scenario_id: str) -> Optional[Dict]:
        """시나리오의 원본 출처 조회"""
        return self.source_mapping.get_source(scenario_id)

    # =========================================================================
    # LLM 프롬프트 템플릿
    # =========================================================================

    @staticmethod
    def get_generation_prompt_template() -> str:
        """
        시나리오 생성용 LLM 프롬프트 템플릿

        Note: LLM 호출은 별도 모듈에서 구현
        """
        return """당신은 경력 10년 이상의 교수설계 전문가입니다.
다음 정보를 바탕으로 일반적인 교수설계 요청 시나리오를 작성해주세요.

## 참고 정보 (직접 인용하지 마세요)
{source_abstract}

## 키워드
{source_keywords}

## 교육 맥락
- 학습자: {learner_age}, {learner_education}, {learner_role}
- 기관: {institution_type}
- 도메인: {education_domain}
- 전달방식: {delivery_mode}
- 규모: {class_size}
- 기술환경: {tech_environment}
- 기간: {duration}

## 난이도
{difficulty}

## 출력 형식 (JSON)
{{
  "title": "교육 과정 제목",
  "context": {{
    "target_audience": "학습 대상",
    "prior_knowledge": "사전 지식",
    "duration": "교육 기간",
    "learning_environment": "학습 환경",
    "class_size": 숫자,
    "additional_context": "추가 맥락"
  }},
  "learning_goals": ["학습목표1", "학습목표2", ...],
  "constraints": {{
    "budget": "low/medium/high",
    "resources": ["리소스1", ...],
    "accessibility": null,
    "language": "ko"
  }},
  "difficulty": "{difficulty}",
  "domain": "도메인"
}}

중요:
- 원본 논문의 제목, 저자, DOI 등을 절대 포함하지 마세요
- 일반적인 교육 상황으로 변환하세요
- 한국어로 작성하세요
"""

    # =========================================================================
    # 시나리오 저장
    # =========================================================================

    def save_scenario(
        self,
        scenario: ScenarioSchema,
        output_dir: str,
    ) -> str:
        """시나리오 JSON 파일 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"scenario_{scenario.scenario_id.lower().replace('-', '_')}.json"
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(scenario.to_json())

        return str(filepath)

    def save_batch(
        self,
        scenarios: List[ScenarioSchema],
        output_dir: str,
        mapping_path: Optional[str] = None,
    ) -> Dict:
        """여러 시나리오 일괄 저장"""
        saved_files = []
        for scenario in scenarios:
            filepath = self.save_scenario(scenario, output_dir)
            saved_files.append(filepath)

        # 출처 매핑 저장
        if mapping_path:
            self.save_source_mapping(mapping_path)

        return {
            "saved_count": len(saved_files),
            "output_dir": output_dir,
            "files": saved_files,
            "mapping_path": mapping_path,
        }

    # =========================================================================
    # 통계
    # =========================================================================

    def summary(self) -> Dict:
        """생성기 상태 요약"""
        return {
            "idld_records": len(self.idld),
            "context_combinations": self.context_matrix.count_all_combinations(),
            "registered_mappings": len(self.source_mapping.mappings),
        }

    def __repr__(self) -> str:
        return f"ScenarioGenerator(idld={len(self.idld)}, mappings={len(self.source_mapping.mappings)})"
