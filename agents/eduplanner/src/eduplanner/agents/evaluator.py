"""
Evaluator Agent: 교수설계 평가 에이전트

ADDIE Rubric 13항목 평가 시스템을 사용하여 교수설계의 품질을 평가합니다.
- Analysis (A1-A3): 학습자/환경/요구 분석
- Design (D1-D3): 목표/평가/전략 설계
- Development (Dev1-Dev2): 자료 개발
- Implementation (I1-I2): 실행 계획
- Evaluation (E1-E3): 평가 도구
"""

import sys
from pathlib import Path
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from eduplanner.agents.base import BaseAgent, AgentConfig
from eduplanner.models.schemas import ADDIEOutput, EvaluationFeedback
from eduplanner.models.skill_tree import LearnerProfile

# ADDIE Rubric 정의 임포트
_project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "evaluator" / "src"))
from isd_evaluator.rubrics.addie_definitions import (
    ADDIE_RUBRIC_DEFINITIONS, DEFAULT_PHASE_WEIGHTS, MAX_SCORE_PER_ITEM
)
from isd_evaluator.models import ADDIEPhase


EVALUATOR_SYSTEM_PROMPT = """당신은 20년 경력의 베테랑 교수설계 전문가입니다.

## 역할
교수설계 산출물을 ADDIE Rubric 13항목 평가 시스템에 따라 체계적으로 평가합니다.

## ADDIE Rubric 평가 기준 (각 항목 0-10점)

### Analysis (분석) - 가중치 25%
- **A1. 학습자 분석의 적절성**: 학습자의 수준, 선지식, 특성, 동기, 요구 등을 파악
- **A2. 수행 맥락 및 환경 분석의 타당성**: 시설, 기기, 시간, 기술 환경 등 조건 분석
- **A3. 요구분석 및 수행 격차 정의의 명확성**: 현재 상태와 목표 상태 간 격차 분석

### Design (설계) - 가중치 25%
- **D1. 학습목표와 요구분석 간 정렬도**: 행동 중심의 목표를 명확히 진술
- **D2. 평가 설계의 타당성 및 정합성**: 학습 목표 달성 여부 판단을 위한 평가 방법
- **D3. 교수전략 및 학습경험 설계의 이론적 적절성**: 적절한 수업 방식, 활동, 매체 활용

### Development (개발) - 가중치 20%
- **Dev1. 프로토타입 개발**: 학습자용 자료, 교수자/운영자 매뉴얼, 평가 도구 개발
- **Dev2. 개발 결과 검토 및 수정**: 전문가 검토를 통한 피드백 반영 및 수정

### Implementation (실행) - 가중치 15%
- **I1. 프로그램 실행 준비**: 오리엔테이션 및 시스템/환경 점검
- **I2. 프로그램 실행**: 프로토타입 실행 및 운영 모니터링

### Evaluation (평가) - 가중치 15%
- **E1. 형성평가**: 파일럿/초기 실행 중 자료 수집 및 1차 개선
- **E2. 총괄평가 및 채택 결정**: 총괄평가 시행, 효과 분석 및 채택 여부 결정
- **E3. 프로그램 개선 및 환류**: 최종 프로그램 개선 및 환류 체계

## 점수 기준 (엄격 적용)
- **9-10점 (매우우수)**: 이론과 실제가 매우 적절히 반영되어 즉시 적용 가능
- **7-8점 (우수)**: 이론과 실제가 적절히 반영되어 적용 가능
- **5-6점 (보통)**: 기본 요소 충족, 구체성/정렬 일부 부족
- **3-4점 (미흡)**: 핵심 요소 부분 결여 또는 실행 가능성 낮음
- **1-2점 (부재)**: 해당 요소 거의 제시되지 않음

## 출력 형식
**반드시 아래 JSON 형식으로만 출력하세요. 다른 텍스트 없이 JSON만 출력하세요.**

```json
{
  "addie_scores": {
    "A1": <0.0-10.0>, "A2": <0.0-10.0>, "A3": <0.0-10.0>,
    "D1": <0.0-10.0>, "D2": <0.0-10.0>, "D3": <0.0-10.0>,
    "Dev1": <0.0-10.0>, "Dev2": <0.0-10.0>,
    "I1": <0.0-10.0>, "I2": <0.0-10.0>,
    "E1": <0.0-10.0>, "E2": <0.0-10.0>, "E3": <0.0-10.0>
  },
  "strengths": [
    "<구체적인 강점 설명>",
    "<구체적인 강점 설명>"
  ],
  "weaknesses": [
    "<구체적인 약점 설명>",
    "<구체적인 약점 설명>"
  ],
  "suggestions": [
    "<구체적인 개선 제안>",
    "<구체적인 개선 제안>"
  ]
}
```

**중요:**
- 각 항목 점수는 **0.0~10.0 사이의 소수점 1자리** (예: 7.5, 8.2, 6.8)
- 세밀한 평가를 위해 **반드시 소수점을 사용**하세요
- strengths, weaknesses, suggestions는 각각 최소 2개, 최대 5개
- JSON 외의 설명이나 텍스트를 포함하지 마세요
"""


class EvaluatorAgent(BaseAgent):
    """교수설계 평가 에이전트"""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            # Evaluator: temperature 0.7로 변경하여 다양한 평가 허용
            config = AgentConfig(
                temperature=0.7,
                max_tokens=4096,
            )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "Evaluator Agent"

    @property
    def role(self) -> str:
        return "ADDIE Rubric 13항목 평가 시스템을 사용하여 교수설계 품질을 평가합니다."

    def run(
        self,
        addie_output: ADDIEOutput,
        learner_profile: Optional[LearnerProfile] = None,
        scenario_context: Optional[str] = None,
    ) -> EvaluationFeedback:
        """
        교수설계 산출물을 평가합니다.

        Args:
            addie_output: ADDIE 5단계 산출물
            learner_profile: 학습자 프로필 (Skill-Tree)
            scenario_context: 시나리오 맥락 정보

        Returns:
            EvaluationFeedback: 평가 결과
        """
        # 프롬프트 구성
        evaluation_prompt = self._build_evaluation_prompt(
            addie_output, learner_profile, scenario_context
        )

        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=evaluation_prompt),
        ]

        # LLM 호출
        response = self.llm.invoke(messages)

        # 응답 파싱
        feedback = self._parse_response(response.content)

        return feedback

    def _build_evaluation_prompt(
        self,
        addie_output: ADDIEOutput,
        learner_profile: Optional[LearnerProfile] = None,
        scenario_context: Optional[str] = None,
    ) -> str:
        """평가 프롬프트 생성"""
        prompt_parts = []

        # 시나리오 맥락
        if scenario_context:
            prompt_parts.append(f"## 시나리오 맥락\n{scenario_context}\n")

        # 학습자 프로필
        if learner_profile:
            prompt_parts.append(learner_profile.skill_tree.to_prompt_context())

        # ADDIE 산출물
        prompt_parts.append("## 평가 대상 교수설계 산출물\n")
        prompt_parts.append(self._format_addie_output(addie_output))

        prompt_parts.append("\n위 교수설계 산출물을 ADDIE Rubric 13항목 기준으로 평가해주세요.")

        return "\n".join(prompt_parts)

    def _format_addie_output(self, addie_output: ADDIEOutput) -> str:
        """ADDIE 산출물을 상세 포맷팅 (개선 사항을 정확히 평가하기 위해)"""
        sections = []

        # Analysis
        analysis = addie_output.analysis
        sections.append("### 1. 분석 (Analysis)")
        sections.append("**학습자 분석:**")
        sections.append(f"- 대상: {analysis.learner_analysis.target_audience}")
        sections.append(f"- 특성: {', '.join(analysis.learner_analysis.characteristics)}")
        sections.append(f"- 사전지식: {analysis.learner_analysis.prior_knowledge}")
        sections.append(f"- 학습 선호: {', '.join(analysis.learner_analysis.learning_preferences)}")
        if analysis.learner_analysis.motivation:
            sections.append(f"- 동기: {analysis.learner_analysis.motivation}")
        sections.append(f"- 예상 어려움: {', '.join(analysis.learner_analysis.challenges)}")

        sections.append("\n**환경 분석:**")
        sections.append(f"- 환경: {analysis.context_analysis.environment}")
        sections.append(f"- 시간: {analysis.context_analysis.duration}")
        sections.append(f"- 제약: {', '.join(analysis.context_analysis.constraints)}")
        sections.append(f"- 자원: {', '.join(analysis.context_analysis.resources)}")

        sections.append("\n**과제 분석:**")
        sections.append(f"- 주요 주제: {', '.join(analysis.task_analysis.main_topics)}")
        sections.append(f"- 세부 주제: {', '.join(analysis.task_analysis.subtopics)}")
        sections.append(f"- 선수 학습: {', '.join(analysis.task_analysis.prerequisites)}")

        # Design
        design = addie_output.design
        sections.append("\n### 2. 설계 (Design)")
        sections.append("**학습 목표:**")
        for obj in design.learning_objectives:
            sections.append(f"- [{obj.id}] [{obj.level}] {obj.statement} (동사: {obj.bloom_verb})")

        sections.append("\n**평가 계획:**")
        sections.append(f"- 진단평가: {', '.join(design.assessment_plan.diagnostic)}")
        sections.append(f"- 형성평가: {', '.join(design.assessment_plan.formative)}")
        sections.append(f"- 총괄평가: {', '.join(design.assessment_plan.summative)}")

        sections.append("\n**교수 전략:**")
        sections.append(f"- 모델: {design.instructional_strategy.model}")
        sections.append(f"- 방법: {', '.join(design.instructional_strategy.methods)}")
        sections.append("- Gagné's 9 Events:")
        for event in design.instructional_strategy.sequence:
            duration_str = f" ({event.duration})" if event.duration else ""
            sections.append(f"  - {event.event}: {event.activity}{duration_str}")

        # Development
        dev = addie_output.development
        sections.append("\n### 3. 개발 (Development)")
        sections.append(f"**레슨 플랜:** {dev.lesson_plan.total_duration}")
        for module in dev.lesson_plan.modules:
            sections.append(f"\n**[모듈] {module.title}** ({module.duration})")
            sections.append(f"  - 목표: {', '.join(module.objectives)}")
            for act in module.activities:
                sections.append(f"  - [{act.time}] {act.activity}: {act.description or ''}")

        sections.append("\n**학습 자료:**")
        for mat in dev.materials:
            sections.append(f"- {mat.type}: {mat.title} - {mat.description or ''}")

        # Implementation
        impl = addie_output.implementation
        sections.append("\n### 4. 실행 (Implementation)")
        sections.append(f"- 전달 방식: {impl.delivery_method}")
        sections.append(f"- 기술 요구사항: {', '.join(impl.technical_requirements)}")
        if impl.facilitator_guide:
            sections.append(f"- 진행자 가이드: {impl.facilitator_guide[:200]}...")
        if impl.learner_guide:
            sections.append(f"- 학습자 가이드: {impl.learner_guide[:200]}...")

        # Evaluation
        eval_section = addie_output.evaluation
        sections.append("\n### 5. 평가 (Evaluation)")
        sections.append(f"**퀴즈 문항 ({len(eval_section.quiz_items)}개):**")
        for item in eval_section.quiz_items[:5]:  # 처음 5개만 표시
            sections.append(f"- [{item.id}] [{item.difficulty or 'N/A'}] {item.question[:100]}")
        if len(eval_section.quiz_items) > 5:
            sections.append(f"  ... 외 {len(eval_section.quiz_items) - 5}개")

        if eval_section.rubric:
            sections.append(f"\n**루브릭 기준:** {', '.join(eval_section.rubric.criteria)}")

        if eval_section.feedback_plan:
            sections.append(f"\n**피드백 계획:** {eval_section.feedback_plan}")

        return "\n".join(sections)

    def _parse_response(self, response_text: str) -> EvaluationFeedback:
        """LLM 응답을 EvaluationFeedback으로 파싱 (ADDIE Rubric 13항목)"""
        import json
        import re

        # ADDIE 13항목 기본값 (각 항목 5.0점)
        addie_items = ["A1", "A2", "A3", "D1", "D2", "D3", "Dev1", "Dev2", "I1", "I2", "E1", "E2", "E3"]
        addie_scores = {item: 5.0 for item in addie_items}
        strengths = []
        weaknesses = []
        suggestions = []

        # 1. JSON 파싱 시도
        json_parsed = False
        try:
            # JSON 블록 추출 (```json ... ``` 또는 직접 JSON)
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            json_str = json_match.group(1) if json_match else response_text.strip()

            # JSON 부분만 추출 (앞뒤 텍스트 제거)
            json_start = json_str.find('{')
            json_end = json_str.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = json_str[json_start:json_end]

            data = json.loads(json_str)

            # addie_scores 추출
            if "addie_scores" in data:
                for item in addie_items:
                    if item in data["addie_scores"]:
                        score_val = float(data["addie_scores"][item])
                        addie_scores[item] = min(10.0, max(0.0, score_val))
                json_parsed = True

            if "strengths" in data and isinstance(data["strengths"], list):
                strengths = data["strengths"][:5]

            if "weaknesses" in data and isinstance(data["weaknesses"], list):
                weaknesses = data["weaknesses"][:5]

            if "suggestions" in data and isinstance(data["suggestions"], list):
                suggestions = data["suggestions"][:5]

        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            pass  # JSON 파싱 실패 시 정규식 폴백으로

        # 2. JSON 파싱 실패 시 정규식 폴백
        if not json_parsed:
            for item in addie_items:
                # 패턴: "A1": 7.5 또는 "A1: 7.5"
                item_match = re.search(
                    rf'"{item}"[:\s]*(\d+(?:\.\d+)?)',
                    response_text,
                    re.IGNORECASE
                )
                if not item_match:
                    item_match = re.search(
                        rf'{item}[:\s]*(\d+(?:\.\d+)?)',
                        response_text,
                        re.IGNORECASE
                    )
                if item_match:
                    parsed_score = float(item_match.group(1))
                    if 0 <= parsed_score <= 10:
                        addie_scores[item] = parsed_score

            # 강점/약점/제안 파싱
            strengths = self._extract_list_items(response_text, ["강점", "Strengths", "strengths"])
            weaknesses = self._extract_list_items(response_text, ["약점", "Weaknesses", "weaknesses"])
            suggestions = self._extract_list_items(response_text, ["제안", "Suggestions", "suggestions"])

        # 가중치 적용 점수 계산
        weighted_score = self._calculate_weighted_score(addie_scores)

        # 총점 계산 (0-100 스케일로 정규화)
        raw_sum = sum(addie_scores.values())  # 최대 130점
        normalized_score = (raw_sum / 130.0) * 100.0

        return EvaluationFeedback(
            score=round(normalized_score, 1),
            strengths=strengths if strengths else ["평가 강점 정보 없음"],
            weaknesses=weaknesses if weaknesses else ["평가 약점 정보 없음"],
            suggestions=suggestions if suggestions else ["개선 제안 정보 없음"],
            addie_scores=addie_scores,
            weighted_score=round(weighted_score, 1),
        )

    def _calculate_weighted_score(self, addie_scores: dict) -> float:
        """ADDIE 단계별 가중치를 적용한 점수 계산"""
        # 단계별 점수 합산
        phase_scores = {
            ADDIEPhase.ANALYSIS: sum(addie_scores.get(k, 0) for k in ["A1", "A2", "A3"]),
            ADDIEPhase.DESIGN: sum(addie_scores.get(k, 0) for k in ["D1", "D2", "D3"]),
            ADDIEPhase.DEVELOPMENT: sum(addie_scores.get(k, 0) for k in ["Dev1", "Dev2"]),
            ADDIEPhase.IMPLEMENTATION: sum(addie_scores.get(k, 0) for k in ["I1", "I2"]),
            ADDIEPhase.EVALUATION: sum(addie_scores.get(k, 0) for k in ["E1", "E2", "E3"]),
        }

        # 단계별 최대 점수
        phase_max = {
            ADDIEPhase.ANALYSIS: 30.0,  # 3항목 * 10점
            ADDIEPhase.DESIGN: 30.0,
            ADDIEPhase.DEVELOPMENT: 20.0,  # 2항목 * 10점
            ADDIEPhase.IMPLEMENTATION: 20.0,  # 2항목 * 10점
            ADDIEPhase.EVALUATION: 30.0,  # 3항목 * 10점
        }

        # 가중치 적용 점수 계산 (0-100 스케일)
        weighted_sum = 0.0
        for phase, raw_score in phase_scores.items():
            normalized = (raw_score / phase_max[phase]) * 100.0
            weighted_sum += normalized * DEFAULT_PHASE_WEIGHTS[phase]

        return weighted_sum

    def _extract_list_items(self, text: str, section_names: list[str]) -> list[str]:
        """텍스트에서 특정 섹션의 리스트 항목 추출"""
        import re
        items = []

        # 섹션 찾기
        for section_name in section_names:
            pattern = rf"{section_name}.*?[:：]\s*\n((?:[-\d\.\s]+[^\n]+\n?)+)"
            section_match = re.search(pattern, text, re.IGNORECASE)
            if section_match:
                section_text = section_match.group(1)
                for line in section_text.strip().split("\n"):
                    # 번호나 대시로 시작하는 항목 추출
                    match = re.match(r"^[\d\.\-\*]+\s*(.+)$", line.strip())
                    if match:
                        items.append(match.group(1).strip())
                break

        return items[:5]  # 최대 5개
