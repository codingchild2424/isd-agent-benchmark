"""
Analyst Agent: 교수설계 분석 에이전트

교수설계 산출물의 오류와 문제점을 분석합니다.
- 논리적 오류 탐지
- 누락 요소 식별
- 일관성 검사
"""

from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from eduplanner.agents.base import BaseAgent, AgentConfig
from eduplanner.models.schemas import (
    ADDIEOutput,
    Analysis,
    Design,
    Development,
    Implementation,
    Evaluation,
    ScenarioInput,
)
from eduplanner.models.skill_tree import LearnerProfile


ANALYST_SYSTEM_PROMPT = """당신은 12년 경력의 교수설계 분석 전문가입니다.

## 역할
교수설계 산출물을 체계적으로 분석하여 오류, 누락, 불일치를 발견합니다.

## 분석 관점
1. 논리적 일관성: 학습 목표와 평가의 정렬, 분석 결과와 설계의 연결
2. 완전성 검사: ADDIE 각 단계 필수 요소, Bloom's Taxonomy, Gagné's 9 Events
3. 학습자 적합성: Skill-Tree 수준 매칭, 인지 부하 적정성
4. 실행 가능성: 시간 배분의 현실성, 자원 요구사항의 타당성

## 출력 형식 (반드시 JSON으로 출력)

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

```json
{
  "quality": "상|중|하",
  "feedback": [
    "가장 심각한 문제점 (1줄)",
    "두 번째 문제점 또는 누락 요소 (1줄)",
    "세 번째 개선 권고 (1줄)"
  ]
}
```

예시:
```json
{
  "quality": "중",
  "feedback": [
    "학습 목표가 Bloom's Taxonomy 동사를 사용하지 않음",
    "평가 문항이 학습 목표와 정렬되지 않음",
    "Gagné 9 Events 중 동기유발 단계 누락"
  ]
}
```
"""


class AnalysisResult:
    """분석 결과"""

    def __init__(
        self,
        quality_level: str = "중",
        summary: str = "",
        errors: list[dict] = None,
        missing_elements: list[dict] = None,
        inconsistencies: list[dict] = None,
        recommendations: list[dict] = None,
    ):
        self.quality_level = quality_level
        self.summary = summary
        self.errors = errors or []
        self.missing_elements = missing_elements or []
        self.inconsistencies = inconsistencies or []
        self.recommendations = recommendations or []

    def has_critical_errors(self) -> bool:
        """치명적 오류 여부"""
        return any(e.get("severity") == "Critical" for e in self.errors)

    def get_high_priority_recommendations(self) -> list[dict]:
        """높은 우선순위 권고사항"""
        return [r for r in self.recommendations if r.get("priority") == "High"]

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "quality_level": self.quality_level,
            "summary": self.summary,
            "errors": self.errors,
            "missing_elements": self.missing_elements,
            "inconsistencies": self.inconsistencies,
            "recommendations": self.recommendations,
        }


class AnalystAgent(BaseAgent):
    """교수설계 분석 에이전트"""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            # Analyst는 균형잡힌 분석을 위해 temperature 0.7 사용
            config = AgentConfig(
                temperature=0.7,
                max_tokens=4096,
            )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "Analyst Agent"

    @property
    def role(self) -> str:
        return "교수설계 산출물의 오류와 문제점을 분석합니다."

    def run(
        self,
        addie_output: ADDIEOutput,
        scenario_input: Optional[ScenarioInput] = None,
        learner_profile: Optional[LearnerProfile] = None,
    ) -> AnalysisResult:
        """
        교수설계 산출물을 분석합니다.

        Args:
            addie_output: ADDIE 산출물
            scenario_input: 원본 시나리오 입력
            learner_profile: 학습자 프로필

        Returns:
            AnalysisResult: 분석 결과
        """
        # 프롬프트 구성
        analysis_prompt = self._build_analysis_prompt(
            addie_output, scenario_input, learner_profile
        )

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=analysis_prompt),
        ]

        # LLM 호출
        response = self.llm.invoke(messages)

        # 응답 파싱
        result = self._parse_response(response.content)

        return result

    def _build_analysis_prompt(
        self,
        addie_output: ADDIEOutput,
        scenario_input: Optional[ScenarioInput] = None,
        learner_profile: Optional[LearnerProfile] = None,
    ) -> str:
        """분석 프롬프트 생성"""
        prompt_parts = []

        # 원본 시나리오
        if scenario_input:
            prompt_parts.append("## 원본 시나리오\n")
            prompt_parts.append(f"**제목:** {scenario_input.title}")
            prompt_parts.append(f"**대상:** {scenario_input.context.target_audience}")
            prompt_parts.append(f"**시간:** {scenario_input.context.duration}")
            prompt_parts.append(f"**환경:** {scenario_input.context.learning_environment}")
            prompt_parts.append(f"**목표:** {', '.join(scenario_input.learning_goals)}\n")

        # 학습자 프로필
        if learner_profile:
            prompt_parts.append(learner_profile.skill_tree.to_prompt_context())

        # ADDIE 산출물
        prompt_parts.append("## 분석 대상 교수설계 산출물\n")
        prompt_parts.append(self._format_addie_output_detailed(addie_output))

        prompt_parts.append("\n위 교수설계 산출물을 체계적으로 분석해주세요.")

        return "\n".join(prompt_parts)

    def _format_addie_output_detailed(self, addie_output: ADDIEOutput) -> str:
        """ADDIE 산출물 상세 포맷팅"""
        sections = []

        # Analysis Phase
        self._format_analysis_phase(sections, addie_output.analysis)

        # Design Phase
        self._format_design_phase(sections, addie_output.design)

        # Development Phase
        self._format_development_phase(sections, addie_output.development)

        # Implementation Phase
        self._format_implementation_phase(sections, addie_output.implementation)

        # Evaluation Phase
        self._format_evaluation_phase(sections, addie_output.evaluation)

        return "\n".join(sections)

    def _format_analysis_phase(self, sections: list, analysis: Analysis) -> None:
        """분석 단계 포맷팅"""
        sections.append("### 1. 분석 (Analysis)")

        # 학습자 분석
        la = analysis.learner_analysis
        sections.append("**학습자 분석:**")
        sections.append(f"  - 대상: {la.target_audience}")
        sections.append(f"  - 특성: {', '.join(la.characteristics) or '미정의'}")
        sections.append(f"  - 사전지식: {la.prior_knowledge or '미정의'}")
        sections.append(f"  - 학습 선호: {', '.join(la.learning_preferences) or '미정의'}")
        sections.append(f"  - 동기: {la.motivation or '미정의'}")
        sections.append(f"  - 예상 어려움: {', '.join(la.challenges) or '미정의'}")

        # 환경 분석
        ca = analysis.context_analysis
        sections.append("\n**환경 분석:**")
        sections.append(f"  - 환경: {ca.environment}")
        sections.append(f"  - 시간: {ca.duration}")
        sections.append(f"  - 제약: {', '.join(ca.constraints) or '미정의'}")
        sections.append(f"  - 자원: {', '.join(ca.resources) or '미정의'}")
        sections.append(f"  - 기술요구: {', '.join(ca.technical_requirements) or '미정의'}")

        # 과제 분석
        ta = analysis.task_analysis
        sections.append("\n**과제 분석:**")
        sections.append(f"  - 주제: {', '.join(ta.main_topics) or '미정의'}")
        sections.append(f"  - 세부: {', '.join(ta.subtopics) or '미정의'}")
        sections.append(f"  - 선수학습: {', '.join(ta.prerequisites) or '미정의'}")

    def _format_design_phase(self, sections: list, design: Design) -> None:
        """설계 단계 포맷팅"""
        sections.append("\n### 2. 설계 (Design)")

        # 학습 목표
        sections.append("**학습 목표:**")
        if design.learning_objectives:
            for obj in design.learning_objectives:
                sections.append(
                    f"  - [{obj.id}] [{obj.level}] {obj.statement} "
                    f"(동사: {obj.bloom_verb}, 측정가능: {obj.measurable})"
                )
        else:
            sections.append("  - (목표 없음)")

        # 평가 계획
        ap = design.assessment_plan
        sections.append("\n**평가 계획:**")
        sections.append(f"  - 진단평가: {', '.join(ap.diagnostic) or '미정의'}")
        sections.append(f"  - 형성평가: {', '.join(ap.formative) or '미정의'}")
        sections.append(f"  - 총괄평가: {', '.join(ap.summative) or '미정의'}")

        # 교수 전략
        ist = design.instructional_strategy
        sections.append("\n**교수 전략:**")
        sections.append(f"  - 모델: {ist.model}")
        sections.append(f"  - 방법: {', '.join(ist.methods) or '미정의'}")
        if ist.sequence:
            sections.append("  - 교수사태:")
            for event in ist.sequence:
                sections.append(f"    * {event.event}: {event.activity}")

    def _format_development_phase(self, sections: list, dev: Development) -> None:
        """개발 단계 포맷팅"""
        sections.append("\n### 3. 개발 (Development)")

        # 레슨 플랜
        sections.append(f"**레슨 플랜:** 총 {dev.lesson_plan.total_duration}")
        if dev.lesson_plan.modules:
            for mod in dev.lesson_plan.modules:
                sections.append(f"  - {mod.title} ({mod.duration})")
                for obj in mod.objectives:
                    sections.append(f"    목표: {obj}")
                for act in mod.activities:
                    sections.append(f"    활동: {act.time} - {act.activity}")
        else:
            sections.append("  - (모듈 없음)")

        # 학습 자료
        sections.append("\n**학습 자료:**")
        if dev.materials:
            for mat in dev.materials:
                details = []
                if mat.slides:
                    details.append(f"슬라이드 {mat.slides}장")
                if mat.duration:
                    details.append(mat.duration)
                if mat.pages:
                    details.append(f"{mat.pages}페이지")
                detail_str = f" ({', '.join(details)})" if details else ""
                sections.append(f"  - [{mat.type}] {mat.title}{detail_str}")
        else:
            sections.append("  - (자료 없음)")

    def _format_implementation_phase(self, sections: list, impl: Implementation) -> None:
        """실행 단계 포맷팅"""
        sections.append("\n### 4. 실행 (Implementation)")
        sections.append(f"**전달 방식:** {impl.delivery_method}")
        sections.append(f"**진행자 가이드:** {impl.facilitator_guide or '미정의'}")
        sections.append(f"**학습자 가이드:** {impl.learner_guide or '미정의'}")
        sections.append(f"**기술 요구사항:** {', '.join(impl.technical_requirements) or '미정의'}")
        sections.append(f"**지원 계획:** {impl.support_plan or '미정의'}")

    def _format_evaluation_phase(self, sections: list, eval_section: Evaluation) -> None:
        """평가 단계 포맷팅"""
        sections.append("\n### 5. 평가 (Evaluation)")

        # 퀴즈 문항
        sections.append(f"**퀴즈 문항:** {len(eval_section.quiz_items)}개")
        for item in eval_section.quiz_items[:3]:  # 최대 3개만 표시
            sections.append(f"  - [{item.type}] {item.question[:50]}...")

        # 루브릭
        if eval_section.rubric:
            sections.append(f"\n**루브릭 기준:** {', '.join(eval_section.rubric.criteria)}")
        else:
            sections.append("\n**루브릭:** 미정의")

        # 피드백 계획
        sections.append(f"**피드백 계획:** {eval_section.feedback_plan or '미정의'}")

    def _parse_response(self, response_text: str) -> AnalysisResult:
        """LLM 응답을 AnalysisResult로 파싱 (JSON 형식)"""
        import json
        import re

        result = AnalysisResult()

        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # ```json 없이 JSON만 있는 경우
            json_str = response_text.strip()

        try:
            data = json.loads(json_str)

            # 품질 수준
            if "quality" in data:
                result.quality_level = data["quality"]

            # 피드백 (3~4줄 리스트)
            if "feedback" in data and isinstance(data["feedback"], list):
                result.summary = "\n".join(f"- {item}" for item in data["feedback"])
                # recommendations에도 저장
                for item in data["feedback"]:
                    result.recommendations.append({
                        "recommendation": item,
                        "priority": "High",
                    })

        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기존 텍스트 파싱 시도
            quality_match = re.search(r"(상|중|하)", response_text)
            if quality_match:
                result.quality_level = quality_match.group(1)
            result.summary = response_text[:500]  # 처음 500자만

        return result

    def _extract_errors(self, text: str) -> list[dict]:
        """오류 목록 추출"""
        import re
        errors = []
        pattern = r"[\d\.\-•]+\s*\[?([^\]]+)\]?\s*-\s*\[?([^\]]+)\]?\s*-\s*\[?(Critical|Major|Minor)\]?"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            errors.append({
                "description": match[0].strip(),
                "location": match[1].strip(),
                "severity": match[2].strip().capitalize(),
            })
        return errors

    def _extract_missing(self, text: str) -> list[dict]:
        """누락 요소 추출"""
        import re
        missing = []
        pattern = r"[\d\.\-•]+\s*\[?([^\]-]+)\]?\s*-\s*\[?([^\]]+)\]?"
        matches = re.findall(pattern, text)
        for match in matches:
            missing.append({
                "element": match[0].strip(),
                "reason": match[1].strip(),
            })
        return missing

    def _extract_inconsistencies(self, text: str) -> list[dict]:
        """불일치 사항 추출"""
        import re
        inconsistencies = []
        pattern = r"[\d\.\-•]+\s*\[?([^\]-]+)\]?\s*-\s*\[?([^\]]+)\]?"
        matches = re.findall(pattern, text)
        for match in matches:
            inconsistencies.append({
                "description": match[0].strip(),
                "related_elements": match[1].strip(),
            })
        return inconsistencies

    def _extract_recommendations(self, text: str) -> list[dict]:
        """권고사항 추출"""
        import re
        recommendations = []
        pattern = r"[\d\.\-•]+\s*\[?([^\]-]+)\]?\s*-\s*\[?(High|Medium|Low)\]?"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            recommendations.append({
                "recommendation": match[0].strip(),
                "priority": match[1].strip().capitalize(),
            })
        return recommendations
