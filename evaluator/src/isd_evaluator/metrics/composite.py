"""
복합 평가 메트릭

ADDIE 루브릭 평가와 궤적 평가를 종합합니다.
"""

from typing import Optional

from isd_evaluator.models import (
    ADDIEPhase, ADDIEScore, CompositeScore, ContextProfile
)
from isd_evaluator.metrics.addie_rubric import ADDIERubricEvaluator
from isd_evaluator.metrics.trajectory import TrajectoryEvaluator
from isd_evaluator.metrics.context_weights import ContextWeightAdjuster
from isd_evaluator.rubrics.addie_definitions import (
    get_item_ids_for_phase, get_rubric_definition, MAX_SCORE_PER_ITEM,
    DEFAULT_PHASE_WEIGHTS
)


class CompositeEvaluator:
    """복합 평가기"""

    def __init__(
        self,
        model: Optional[str] = None,
        output_weight: float = 0.7,
        trajectory_weight: float = 0.3,
        use_llm: bool = True,
        use_context_weights: bool = True,
        temperature: float = 0.0,
    ):
        """
        Args:
            model: LLM 모델명 (None이면 환경변수 또는 기본값 사용)
            output_weight: 산출물 평가 가중치 (0-1)
            trajectory_weight: 궤적 평가 가중치 (0-1)
            use_llm: ADDIE 평가에 LLM 사용 여부
            use_context_weights: 컨텍스트 기반 가중치 조정 사용 여부
            temperature: LLM temperature (0.0=결정적, 0.7=일반적)
        """
        self.output_weight = output_weight
        self.trajectory_weight = trajectory_weight
        self.use_llm = use_llm
        self.use_context_weights = use_context_weights
        self.temperature = temperature

        self.addie_evaluator = ADDIERubricEvaluator(model=model, temperature=temperature)
        self.trajectory_evaluator = TrajectoryEvaluator()

    def evaluate(
        self,
        addie_output: dict,
        scenario: Optional[dict] = None,
        trajectory: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> CompositeScore:
        """
        Agent 산출물과 궤적을 종합 평가합니다.

        Args:
            addie_output: ADDIE 산출물
            scenario: 원본 시나리오 (선택)
            trajectory: 궤적 (선택)
            metadata: 메타데이터 (선택)

        Returns:
            CompositeScore: 종합 평가 점수
        """
        # 컨텍스트 기반 가중치 조정
        context_profile = None
        if self.use_context_weights and scenario:
            adjuster = ContextWeightAdjuster.from_scenario(scenario)
            context_profile = adjuster.context_profile
            adjusted_weights = adjuster.get_adjusted_weights()
            self.addie_evaluator.phase_weights = adjusted_weights

        # ADDIE 평가
        if self.use_llm:
            addie_score = self.addie_evaluator.evaluate(addie_output, scenario)
        else:
            addie_score = self._evaluate_addie_rule_based(addie_output, scenario)

        # 궤적 평가 (제공된 경우)
        trajectory_score = None
        if trajectory:
            trajectory_score = self.trajectory_evaluator.evaluate(trajectory, metadata)

        return CompositeScore(
            addie=addie_score,
            trajectory=trajectory_score,
            addie_weight=self.output_weight,
            trajectory_weight=self.trajectory_weight,
            context_profile=context_profile,
        )

    def _evaluate_addie_rule_based(
        self,
        addie_output: dict,
        scenario: Optional[dict] = None,
    ) -> ADDIEScore:
        """규칙 기반 ADDIE 평가 (LLM 미사용)"""
        from isd_evaluator.models import RubricItem, PhaseScore

        phase_weights = self.addie_evaluator.phase_weights

        phase_scores = {}

        for phase in ADDIEPhase:
            item_ids = get_item_ids_for_phase(phase)
            phase_data = addie_output.get(phase.value, {})

            items = []
            for item_id in item_ids:
                item_def = get_rubric_definition(phase.value, item_id)
                score = self._calculate_rule_score(phase, item_id, phase_data)

                items.append(RubricItem(
                    item_id=item_id,
                    phase=phase,
                    name=item_def.get("name", item_id),
                    description=item_def.get("description", ""),
                    score=score,
                    reasoning="규칙 기반 평가",
                ))

            raw_score = sum(item.score for item in items)
            max_score = len(items) * MAX_SCORE_PER_ITEM

            phase_scores[phase] = PhaseScore(
                phase=phase,
                items=items,
                raw_score=raw_score,
                weighted_score=raw_score * phase_weights[phase],
                max_score=max_score,
            )

        # 종합 점수 계산
        total_raw = sum(ps.raw_score for ps in phase_scores.values())
        total_weighted = sum(
            ps.raw_score * phase_weights[phase]
            for phase, ps in phase_scores.items()
        )
        max_possible = sum(
            ps.max_score * phase_weights[phase]
            for phase, ps in phase_scores.items()
        )
        normalized = (total_weighted / max_possible) * 100 if max_possible > 0 else 0

        return ADDIEScore(
            analysis=phase_scores[ADDIEPhase.ANALYSIS],
            design=phase_scores[ADDIEPhase.DESIGN],
            development=phase_scores[ADDIEPhase.DEVELOPMENT],
            implementation=phase_scores[ADDIEPhase.IMPLEMENTATION],
            evaluation=phase_scores[ADDIEPhase.EVALUATION],
            total_raw=total_raw,
            total_weighted=total_weighted,
            normalized_score=normalized,
            strengths=[],
            improvements=[],
            overall_assessment="규칙 기반 평가",
        )

    def _calculate_rule_score(
        self,
        phase: ADDIEPhase,
        item_id: str,
        phase_data: dict,
    ) -> float:
        """규칙 기반 개별 항목 점수 계산"""
        base_score = 5.0

        if not phase_data:
            return 2.0

        # 데이터 존재 여부에 따른 기본 점수
        content = str(phase_data)
        content_length = len(content)

        # 내용 길이에 따른 점수
        if content_length < 100:
            base_score = 3.0
        elif content_length < 500:
            base_score = 5.0
        elif content_length < 1000:
            base_score = 6.0
        elif content_length < 2000:
            base_score = 7.0
        else:
            base_score = 8.0

        # 키워드 기반 보너스
        keywords = {
            ADDIEPhase.ANALYSIS: ["학습자", "분석", "환경", "요구", "격차", "특성"],
            ADDIEPhase.DESIGN: ["목표", "평가", "전략", "정렬", "설계"],
            ADDIEPhase.DEVELOPMENT: ["자료", "예시", "피드백", "매체", "콘텐츠"],
            ADDIEPhase.IMPLEMENTATION: ["운영", "계획", "가이드", "지원", "실행"],
            ADDIEPhase.EVALUATION: ["평가", "타당도", "개선", "형성", "피드백"],
        }

        phase_keywords = keywords.get(phase, [])
        keyword_count = sum(1 for kw in phase_keywords if kw in content)
        keyword_bonus = min(keyword_count * 0.3, 1.5)

        return min(10.0, base_score + keyword_bonus)

    def compare_agents(
        self,
        results: list[dict],
        scenario: Optional[dict] = None,
    ) -> dict:
        """
        여러 Agent의 결과를 비교 평가합니다.

        Args:
            results: Agent 결과 목록 [{agent_id, addie_output, trajectory, metadata}, ...]
            scenario: 공통 시나리오

        Returns:
            비교 결과
        """
        evaluations = []

        for result in results:
            agent_id = result.get("agent_id", "unknown")
            addie_output = result.get("addie_output", {})
            trajectory = result.get("trajectory")
            metadata = result.get("metadata")

            score = self.evaluate(
                addie_output=addie_output,
                scenario=scenario,
                trajectory=trajectory,
                metadata=metadata,
            )

            evaluations.append({
                "agent_id": agent_id,
                "score": score,
            })

        # 순위 정렬
        evaluations.sort(key=lambda x: x["score"].total, reverse=True)

        return {
            "rankings": [
                {
                    "rank": i + 1,
                    "agent_id": e["agent_id"],
                    "total_score": e["score"].total,
                    "addie_score": e["score"].addie_score,
                    "trajectory_score": e["score"].trajectory_score,
                    "trajectory_details": (
                        e["score"].trajectory.to_dict()
                        if e["score"].trajectory else None
                    ),
                    "phase_scores": {
                        phase.value: ps.percentage
                        for phase, ps in e["score"].addie.phases.items()
                    },
                    "details": e["score"].addie.to_dict(),
                }
                for i, e in enumerate(evaluations)
            ],
            "best_agent": evaluations[0]["agent_id"] if evaluations else None,
            "evaluations": evaluations,
        }
