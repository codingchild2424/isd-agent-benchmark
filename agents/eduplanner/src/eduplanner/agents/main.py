"""
EduPlanner Main Agent: 3-Agent 협업 오케스트레이터

Evaluator, Optimizer, Analyst의 협업을 조율하여
교수설계 산출물을 생성하고 개선합니다.

협업 흐름:
1. Generator가 초기 ADDIE 산출물 생성
2. Evaluator가 CIDPP 평가 수행
3. 목표 점수 미달 시:
   - Analyst가 문제점 분석
   - Optimizer가 개선 수행
4. 반복하여 품질 목표 달성
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from eduplanner.agents.base import BaseAgent, AgentConfig
from eduplanner.agents.evaluator import EvaluatorAgent
from eduplanner.agents.optimizer import OptimizerAgent
from eduplanner.agents.analyst import AnalystAgent
from eduplanner.models.schemas import (
    ScenarioInput,
    ADDIEOutput,
    AgentResult,
    Trajectory,
    Metadata,
    ToolCall,
    Analysis,
    Design,
    Development,
    Implementation,
    Evaluation,
    LearnerAnalysis,
    ContextAnalysis,
    TaskAnalysis,
    NeedsAnalysis,
    LearningObjective,
    AssessmentPlan,
    InstructionalStrategy,
    InstructionalEvent,
    PrototypeDesign,
    LessonPlan,
    Module,
    Activity,
    Material,
    SlideContent,
    QuizItem,
    Rubric,
)
from eduplanner.models.skill_tree import LearnerProfile
from eduplanner.agents.prompts import (
    ANALYSIS_PROMPT,
    DESIGN_PROMPT,
    DEVELOPMENT_PROMPT,
    IMPLEMENTATION_PROMPT,
    EVALUATION_PROMPT,
)


# 기존 통합 프롬프트 (Optimizer에서 사용 - 하위 호환성 유지용)
GENERATOR_SYSTEM_PROMPT = """당신은 20년 경력의 교수설계 전문가입니다.

## 역할
ADDIE 모델에 따라 체계적이고 **상세한** 교수설계 산출물을 생성합니다.

## ⚠️ 중요: 최소 요구사항 (MINIMUM REQUIREMENTS)

아래 요구사항을 반드시 충족해야 합니다. 미충족 시 품질 기준 미달로 평가됩니다.

### Analysis 단계
- learner_analysis.characteristics: **최소 5개** 구체적 특성 (각 1문장 이상)
- learner_analysis.learning_preferences: **최소 4개**
- learner_analysis.challenges: **최소 3개** 예상 어려움
- learner_analysis.motivation: **2-3문장**으로 동기 수준과 이유 설명
- context_analysis.constraints: **최소 3개**
- context_analysis.resources: **최소 3개**
- context_analysis.technical_requirements: **최소 2개**
- task_analysis.main_topics: **최소 3개**
- task_analysis.subtopics: **최소 6개** (각 main_topic당 최소 2개)
- task_analysis.prerequisites: **최소 2개**

### Design 단계
- learning_objectives: **최소 5개** (Bloom's 수준 분산 필수)
  - 기억/이해: 1-2개
  - 적용/분석: 2-3개
  - 평가/창조: 1-2개
- assessment_plan.diagnostic: **최소 2개** 방법
- assessment_plan.formative: **최소 2개** 방법
- assessment_plan.summative: **최소 2개** 방법
- instructional_strategy.sequence: **9개 Event 모두** 포함 (필수!)
- instructional_strategy.methods: **최소 3개**

### Development 단계
- lesson_plan.modules: **최소 3개** 모듈
- 각 module.activities: **최소 3개** 활동
- materials: **최소 5개** 자료 (slides, pages 값 필수 입력 - null 금지)
  - **content 필드 필수**: 각 자료의 실제 내용을 작성
  - 유인물: 실제 배포할 텍스트 내용 (최소 500자)
  - 퀴즈 자료: 문항과 선택지 포함
  - **프레젠테이션/슬라이드 자료의 경우 slide_contents 필수** (반드시 포함!):
    - 각 슬라이드별 상세 콘텐츠 (slide_number, title, bullet_points, speaker_notes)
    - 슬라이드당 3-5개의 핵심 bullet_points
    - speaker_notes에 발표자를 위한 상세 설명 포함
    - 예시 형식:
    ```json
    {
      "type": "프레젠테이션",
      "title": "강의 슬라이드",
      "slides": 10,
      "slide_contents": [
        {"slide_number": 1, "title": "교육 소개", "bullet_points": ["환영 인사", "학습 목표", "일정 안내"], "speaker_notes": "참가자들을 환영하며 교육 목표를 안내합니다."},
        {"slide_number": 2, "title": "핵심 개념", "bullet_points": ["개념 1 설명", "개념 2 설명", "실제 사례"], "speaker_notes": "핵심 개념을 예시와 함께 설명합니다."}
      ]
    }
    ```

### Evaluation 단계
- quiz_items: **최소 10개** (난이도별 분산)
  - easy: 3-4개
  - medium: 4-5개
  - hard: 2-3개
  - **options 필수**: 객관식의 경우 4개 선택지 제공
  - **answer 필수**: 정답 명시
  - **explanation 필수**: 정답 해설 제공

### Implementation 단계
- facilitator_guide: **최소 200자** 상세 가이드 (구체적인 진행 지침)
- learner_guide: **최소 200자** 상세 가이드 (학습 방법 안내)
- technical_requirements: **최소 2개**

- rubric.criteria: **최소 5개** 평가 기준
- rubric.levels: 각 수준(excellent/good/needs_improvement)별 **구체적 기준** 명시 (각 1-2문장)
- feedback_plan: **2-3문장** 상세 계획

## ADDIE 프레임워크

### 1. 분석 (Analysis)
- 학습자 분석: 대상, 특성, 사전지식, 선호도, 동기, 어려움
- 환경 분석: 학습환경, 시간, 제약, 자원, 기술요구사항
- 과제 분석: 주요주제, 세부주제, 선수학습

### 2. 설계 (Design)
- 학습 목표: Bloom's Taxonomy 수준별 목표 설정
- 평가 계획: 진단/형성/총괄 평가 방법
- 교수 전략: Gagné's 9 Events 기반 교수사태

### 3. 개발 (Development)
- 레슨 플랜: 모듈별 구성, 시간 배분
- 학습 자료: 필요한 교재, 슬라이드, 미디어

### 4. 실행 (Implementation)
- 전달 방식, 진행자 가이드, 학습자 가이드
- 기술 요구사항, 지원 계획

### 5. 평가 (Evaluation)
- 퀴즈 문항, 평가 루브릭, 피드백 계획

## Bloom's Taxonomy 동사
- 기억: 정의하다, 나열하다, 인식하다, 회상하다, 명명하다
- 이해: 설명하다, 요약하다, 해석하다, 분류하다, 예시하다
- 적용: 적용하다, 시연하다, 사용하다, 실행하다, 구현하다
- 분석: 분석하다, 비교하다, 구별하다, 조직하다, 귀인하다
- 평가: 평가하다, 판단하다, 비평하다, 정당화하다, 검증하다
- 창조: 설계하다, 개발하다, 생성하다, 구성하다, 계획하다

## Gagné's 9 Events (반드시 모두 포함!)
1. 주의 획득 (Gain attention)
2. 학습 목표 제시 (Inform learners of objectives)
3. 선수 학습 상기 (Stimulate recall of prior learning)
4. 학습 내용 제시 (Present content)
5. 학습 안내 제공 (Provide learning guidance)
6. 연습 유도 (Elicit performance)
7. 피드백 제공 (Provide feedback)
8. 수행 평가 (Assess performance)
9. 파지 및 전이 강화 (Enhance retention and transfer)

## 📋 품질 자가검증 (생성 후 반드시 확인)

산출물 생성 후 다음을 확인하세요:
□ 모든 learning_objectives가 측정 가능한 동사로 시작하는가?
□ instructional_strategy.sequence가 9개 Event를 모두 포함하는가?
□ 각 quiz_item이 특정 learning_objective와 연결(objective_id)되는가?
□ 전체 시간 배분 합계가 duration과 일치하는가?
□ 모든 설명이 최소 2문장 이상인가?
□ materials의 slides, pages 값이 모두 숫자로 입력되었는가?
□ **facilitator_guide가 200자 이상이며 단계별 지침을 포함하는가?**
□ **learner_guide가 100자 이상이며 구체적인 학습 안내를 포함하는가?**

## ⚠️ Implementation 가이드 필수 요소

**facilitator_guide** (최소 200자):
- 단계별 진행 지침 (1, 2, 3... 번호 사용)
- 각 모듈/활동별 구체적 안내
- 시간 관리 팁
- 학습자 참여 유도 방법

예시:
"1. 주의 집중 (5분): 동영상 재생 후 간단한 질문으로 관심 유도
2. 목표 제시 (3분): 화이트보드에 학습 목표를 시각적으로 제시
3. 내용 전달 (20분): 슬라이드를 보며 핵심 개념 설명..."

**learner_guide** (최소 100자):
- 학습 전/중/후 활동 안내
- 참여 방법
- 질문/도움 요청 방법

## 출력 예시 (상세도 참고용 - 내용은 시나리오에 맞게 작성)

아래는 **상세도 수준**의 예시입니다. 실제 내용은 주어진 시나리오에 맞게 작성하세요:

```json
{
  "analysis": {
    "learner_analysis": {
      "target_audience": "[시나리오의 대상자]",
      "characteristics": [
        "[대상자의 연령대, 배경 등 인구통계학적 특성]",
        "[교육 수준 및 전문 분야]",
        "[관련 경험 수준]",
        "[학습 태도 및 성향]",
        "[기술/도구 활용 능력]"
      ],
      "prior_knowledge": "[대상자가 이미 알고 있는 내용과 부족한 부분을 2-3문장으로 상세히 기술]",
      "learning_preferences": [
        "[선호하는 학습 방식 1]",
        "[선호하는 학습 방식 2]",
        "[선호하는 콘텐츠 유형]",
        "[학습 환경 선호도]"
      ],
      "motivation": "[학습 동기를 2-3문장으로 구체적으로 기술. 내적 동기와 외적 동기 모두 포함]",
      "challenges": [
        "[예상되는 학습 어려움 1]",
        "[예상되는 학습 어려움 2]",
        "[예상되는 학습 어려움 3]"
      ]
    },
    "context_analysis": {
      "environment": "[학습 환경]",
      "duration": "[총 학습 시간]",
      "constraints": ["[제약조건 1]", "[제약조건 2]", "[제약조건 3]"],
      "resources": ["[가용 자원 1]", "[가용 자원 2]", "[가용 자원 3]", "[가용 자원 4]"],
      "technical_requirements": ["[기술 요구사항 1]", "[기술 요구사항 2]", "[기술 요구사항 3]"]
    },
    "task_analysis": {
      "main_topics": ["[주요 주제 1]", "[주요 주제 2]", "[주요 주제 3]"],
      "subtopics": ["[세부 주제 1-1]", "[세부 주제 1-2]", "[세부 주제 2-1]", "[세부 주제 2-2]", "[세부 주제 3-1]", "[세부 주제 3-2]"],
      "prerequisites": ["[선수 학습 1]", "[선수 학습 2]"]
    }
  },
  "design": {
    "learning_objectives": [
      {"id": "OBJ-01", "level": "기억", "statement": "[측정 가능한 행동 동사로 시작하는 목표]", "bloom_verb": "[동사]", "measurable": true},
      {"id": "OBJ-02", "level": "이해", "statement": "[측정 가능한 행동 동사로 시작하는 목표]", "bloom_verb": "[동사]", "measurable": true},
      {"id": "OBJ-03", "level": "적용", "statement": "[측정 가능한 행동 동사로 시작하는 목표]", "bloom_verb": "[동사]", "measurable": true},
      {"id": "OBJ-04", "level": "적용", "statement": "[측정 가능한 행동 동사로 시작하는 목표]", "bloom_verb": "[동사]", "measurable": true},
      {"id": "OBJ-05", "level": "분석", "statement": "[측정 가능한 행동 동사로 시작하는 목표]", "bloom_verb": "[동사]", "measurable": true}
    ],
    "assessment_plan": {
      "diagnostic": ["[진단 평가 방법 1]", "[진단 평가 방법 2]"],
      "formative": ["[형성 평가 방법 1]", "[형성 평가 방법 2]", "[형성 평가 방법 3]"],
      "summative": ["[총괄 평가 방법 1]", "[총괄 평가 방법 2]"]
    },
    "instructional_strategy": {
      "model": "Gagné's 9 Events",
      "sequence": [
        {"event": "주의 획득", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]},
        {"event": "학습 목표 제시", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]},
        {"event": "선수 학습 상기", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]},
        {"event": "학습 내용 제시", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]},
        {"event": "학습 안내 제공", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]},
        {"event": "연습 유도", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]},
        {"event": "피드백 제공", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]},
        {"event": "수행 평가", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]},
        {"event": "파지 및 전이 강화", "activity": "[구체적 활동 설명]", "duration": "[시간]", "resources": ["[자원]"]}
      ],
      "methods": ["[교수 방법 1]", "[교수 방법 2]", "[교수 방법 3]"]
    }
  },
  "implementation": {
    "delivery_method": "[전달 방식]",
    "facilitator_guide": "1. 사전 준비 (10분 전): 강의실 점검, 프로젝터 테스트, 학습 자료 배치\n2. 오프닝 (5분): 환영 인사, 오늘의 학습 목표 안내, 아이스브레이킹 활동\n3. 모듈 1 진행 (20분): 슬라이드 설명 후 그룹 토론, 질의응답\n4. 실습 지도 (15분): 개별 실습 지원, 어려워하는 학습자 1:1 도움\n5. 마무리 (5분): 핵심 내용 요약, 다음 단계 안내",
    "learner_guide": "1. 학습 전: 사전 설문 작성, 개인 학습 목표 설정\n2. 학습 중: 적극적 질문, 그룹 활동 참여, 실습 시 동료와 협력\n3. 학습 후: 핸드아웃 복습, 업무 적용 계획 수립",
    "technical_requirements": ["[기술 요구사항 1]", "[기술 요구사항 2]", "[기술 요구사항 3]"],
    "support_plan": "[학습자 지원 계획]"
  }
}
```

## 출력 형식
위 예시의 상세도를 참고하여 **시나리오에 맞는** 완전한 JSON 구조로 출력하세요:

```json
{
  "analysis": {...},
  "design": {...},
  "development": {...},
  "implementation": {...},
  "evaluation": {...}
}
```
"""


class EduPlannerAgent(BaseAgent):
    """EduPlanner 메인 에이전트 (3-Agent 오케스트레이터)"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        max_iterations: int = 2,
        target_score: float = 90.0,
        debug: bool = False,
    ):
        if config is None:
            config = AgentConfig(
                temperature=0.7,
                max_tokens=16384,
            )
        super().__init__(config)

        self.max_iterations = max_iterations
        self.target_score = target_score
        self.debug = debug

        # 하위 에이전트들
        self._evaluator: Optional[EvaluatorAgent] = None
        self._optimizer: Optional[OptimizerAgent] = None
        self._analyst: Optional[AnalystAgent] = None

    @property
    def name(self) -> str:
        return "EduPlanner Agent"

    @property
    def role(self) -> str:
        return "3-Agent 협업을 통해 고품질 교수설계 산출물을 생성합니다."

    @property
    def evaluator(self) -> EvaluatorAgent:
        """Evaluator Agent (지연 초기화)"""
        if self._evaluator is None:
            self._evaluator = EvaluatorAgent()
        return self._evaluator

    @property
    def optimizer(self) -> OptimizerAgent:
        """Optimizer Agent (지연 초기화)"""
        if self._optimizer is None:
            self._optimizer = OptimizerAgent(debug=self.debug)
        return self._optimizer

    @property
    def analyst(self) -> AnalystAgent:
        """Analyst Agent (지연 초기화)"""
        if self._analyst is None:
            self._analyst = AnalystAgent()
        return self._analyst

    def run(self, scenario_input: ScenarioInput) -> AgentResult:
        """
        교수설계 시나리오를 입력받아 ADDIE 산출물을 생성합니다.

        Args:
            scenario_input: 시나리오 입력

        Returns:
            AgentResult: 최종 결과 (ADDIE 산출물 + 궤적 + 메타데이터)
        """
        start_time = datetime.now()
        trajectory = Trajectory()
        total_tokens = 0
        step_counter = 0  # tool_calls step 카운터

        # target_audience 추론: 기존 필드가 없으면 IDLD 필드에서 조합
        target_audience = scenario_input.context.target_audience
        if not target_audience:
            parts_audience = []
            if scenario_input.context.learner_age:
                parts_audience.append(scenario_input.context.learner_age)
            if scenario_input.context.learner_role:
                parts_audience.append(scenario_input.context.learner_role)
            if scenario_input.context.learner_education:
                parts_audience.append(f"({scenario_input.context.learner_education})")
            target_audience = " ".join(parts_audience) if parts_audience else "학습자"

        # 학습자 프로필 생성
        learner_profile = LearnerProfile.from_scenario(
            target_audience=target_audience,
            prior_knowledge=scenario_input.context.prior_knowledge,
            learning_environment=scenario_input.context.learning_environment or "미정",
        )

        # 시나리오 컨텍스트 문자열
        scenario_context = self._build_scenario_context(scenario_input)

        # 1단계: 초기 ADDIE 산출물 생성
        trajectory.reasoning_steps.append("Step 1: 초기 ADDIE 산출물 생성")
        gen_start = datetime.now()
        addie_output = self._generate_initial_output(scenario_input, learner_profile)
        gen_end = datetime.now()
        step_counter += 1

        trajectory.tool_calls.append(ToolCall(
            step=step_counter,
            tool="generate_initial_addie",
            args={"scenario_id": scenario_input.scenario_id},
            result="ADDIE 산출물 초기 생성 완료",
            timestamp=gen_start,
            duration_ms=int((gen_end - gen_start).total_seconds() * 1000),
        ))

        # 2단계: 반복적 개선 루프
        best_output = addie_output
        best_score = 0.0
        score_history = []  # 점수 이력 추적

        for iteration in range(1, self.max_iterations + 1):
            trajectory.reasoning_steps.append(f"Step {iteration + 1}: 평가 및 개선 반복 {iteration}")

            # 2.1 & 2.2: Evaluator와 Analyst 병렬 실행 (#79 성능 최적화)
            parallel_start = datetime.now()

            # ThreadPoolExecutor로 동기 함수들을 병렬 실행
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Evaluator 실행
                eval_future = executor.submit(
                    self.evaluator.run,
                    addie_output=addie_output,
                    learner_profile=learner_profile,
                    scenario_context=scenario_context,
                )
                # Analyst 실행
                analyst_future = executor.submit(
                    self.analyst.run,
                    addie_output=addie_output,
                    scenario_input=scenario_input,
                    learner_profile=learner_profile,
                )

                # 결과 수집
                feedback = eval_future.result()
                analysis_result = analyst_future.result()

            parallel_end = datetime.now()

            # Evaluator 결과 기록
            step_counter += 1
            trajectory.tool_calls.append(ToolCall(
                step=step_counter,
                tool="evaluate_addie",
                args={"iteration": iteration},
                result=f"ADDIE 평가 완료: {feedback.score:.1f}점",
                timestamp=parallel_start,
                duration_ms=int((parallel_end - parallel_start).total_seconds() * 1000 / 2),
                output_data={"score": feedback.score, "addie": feedback.addie_scores},
                feedback={"suggestions": feedback.suggestions[:2]} if feedback.suggestions else None,
            ))

            # Analyst 결과 기록
            step_counter += 1
            trajectory.tool_calls.append(ToolCall(
                step=step_counter,
                tool="analyze_addie",
                args={"iteration": iteration},
                result=f"분석 완료: 품질={analysis_result.quality_level}, 오류={len(analysis_result.errors)}개",
                timestamp=parallel_start,
                duration_ms=int((parallel_end - parallel_start).total_seconds() * 1000 / 2),
                output_data={
                    "quality": analysis_result.quality_level,
                    "errors": len(analysis_result.errors),
                    "missing": len(analysis_result.missing_elements),
                },
                feedback={"summary": analysis_result.summary} if hasattr(analysis_result, 'summary') and analysis_result.summary else None,
            ))

            # 점수 이력 추가
            score_history.append(feedback.score)

            # 최고 점수 갱신
            if feedback.score > best_score:
                best_score = feedback.score
                best_output = addie_output

            # 조기 종료 조건 (#73 성능 최적화)
            # 1. 목표 점수(85점) 도달 시 종료
            # 2. 점수 개선 없으면 종료 (반복 낭비 방지)
            if feedback.score >= 85.0:
                trajectory.reasoning_steps.append(
                    f"조기 종료: 목표 점수 달성 ({feedback.score:.1f}점)"
                )
                break
            if len(score_history) >= 2 and feedback.score <= score_history[-2]:
                trajectory.reasoning_steps.append(
                    f"조기 종료: 점수 개선 없음 ({score_history[-2]:.1f} → {feedback.score:.1f})"
                )
                break

            # 2.3: 최적화 (Evaluator feedback + Analyst 분석 결과 모두 전달)
            opt_start = datetime.now()
            addie_output = self.optimizer.run(
                addie_output=addie_output,
                feedback=feedback,
                analysis_result=analysis_result,
                learner_profile=learner_profile,
                scenario_context=scenario_context,
            )
            opt_end = datetime.now()
            step_counter += 1

            trajectory.tool_calls.append(ToolCall(
                step=step_counter,
                tool="optimize_addie",
                args={"iteration": iteration},
                result="ADDIE 산출물 최적화 완료",
                timestamp=opt_start,
                duration_ms=int((opt_end - opt_start).total_seconds() * 1000),
                output_data={"optimized": True, "feedback_score": feedback.score},
                feedback={"improvement_areas": feedback.weaknesses[:3]} if feedback.weaknesses else None,
            ))

        # 최고 점수 버전으로 복원 (점수가 떨어진 경우 대비)
        addie_output = best_output
        trajectory.reasoning_steps.append(
            f"최고 점수 버전 사용: {best_score:.1f}점"
        )

        # 최종 평가
        final_eval_start = datetime.now()
        final_feedback = self.evaluator.run(
            addie_output=addie_output,
            learner_profile=learner_profile,
            scenario_context=scenario_context,
        )
        final_eval_end = datetime.now()
        step_counter += 1

        trajectory.tool_calls.append(ToolCall(
            step=step_counter,
            tool="final_evaluate_addie",
            args={"type": "final"},
            result=f"최종 CIDPP 평가: {final_feedback.score:.1f}점",
            timestamp=final_eval_start,
            duration_ms=int((final_eval_end - final_eval_start).total_seconds() * 1000),
        ))

        trajectory.reasoning_steps.append(
            f"최종 점수: {final_feedback.score:.1f}/100"
        )

        # 메타데이터 생성
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        metadata = Metadata(
            model=self.config.model,
            total_tokens=total_tokens,
            execution_time_seconds=execution_time,
            agent_version="0.1.0",
            iterations=len([
                tc for tc in trajectory.tool_calls
                if tc.tool == "optimize_addie"
            ]),
        )

        # Final fallback: slide_contents가 없는 경우 자동 생성
        if addie_output and addie_output.development:
            materials = addie_output.development.materials or []
            # modules는 development.lesson_plan에 있음
            lesson_plan = addie_output.development.lesson_plan
            modules = lesson_plan.modules if lesson_plan else []
            has_slide_contents = any(
                mat.slide_contents for mat in materials if mat
            )
            if not has_slide_contents and modules:
                # learning_objectives 가져오기
                learning_objectives = addie_output.design.learning_objectives if addie_output.design else []
                fallback_slides = self._generate_fallback_slides(modules, scenario_input, learning_objectives)
                if fallback_slides:
                    from eduplanner.models.schemas import Material
                    new_material = Material(
                        type="프레젠테이션",
                        title="교육 슬라이드",
                        description="모듈 정보 기반 자동 생성 슬라이드",
                        slides=len(fallback_slides),
                        slide_contents=fallback_slides,
                    )
                    addie_output.development.materials.append(new_material)

        return AgentResult(
            scenario_id=scenario_input.scenario_id,
            agent_id="eduplanner",
            timestamp=end_time,
            addie_output=addie_output,
            trajectory=trajectory,
            metadata=metadata,
        )

    def _build_scenario_context(self, scenario_input: ScenarioInput) -> str:
        """시나리오 컨텍스트 문자열 생성"""
        # target_audience 추론: 기존 필드가 없으면 IDLD 필드에서 조합
        target_audience = scenario_input.context.target_audience
        if not target_audience:
            # IDLD 시나리오 필드에서 조합
            parts_audience = []
            if scenario_input.context.learner_age:
                parts_audience.append(scenario_input.context.learner_age)
            if scenario_input.context.learner_role:
                parts_audience.append(scenario_input.context.learner_role)
            if scenario_input.context.learner_education:
                parts_audience.append(f"({scenario_input.context.learner_education})")
            target_audience = " ".join(parts_audience) if parts_audience else "학습자"

        parts = [
            f"**제목:** {scenario_input.title}",
            f"**대상:** {target_audience}",
            f"**시간:** {scenario_input.context.duration or '미정'}",
            f"**환경:** {scenario_input.context.learning_environment or '미정'}",
            f"**목표:** {', '.join(scenario_input.learning_goals)}",
        ]

        if scenario_input.context.prior_knowledge:
            parts.append(f"**사전지식:** {scenario_input.context.prior_knowledge}")

        if scenario_input.context.class_size:
            parts.append(f"**학습자 수:** {scenario_input.context.class_size}")

        if scenario_input.context.institution_type:
            parts.append(f"**기관 유형:** {scenario_input.context.institution_type}")

        if scenario_input.constraints:
            if scenario_input.constraints.budget:
                parts.append(f"**예산:** {scenario_input.constraints.budget}")
            if scenario_input.constraints.resources:
                parts.append(f"**자원:** {', '.join(scenario_input.constraints.resources)}")
            if scenario_input.constraints.tech_requirements:
                parts.append(f"**기술요건:** {scenario_input.constraints.tech_requirements}")

        return "\n".join(parts)

    def _generate_initial_output(
        self,
        scenario_input: ScenarioInput,
        learner_profile: LearnerProfile,
    ) -> ADDIEOutput:
        """
        ADDIE 산출물을 순차적 파이프라인으로 생성

        각 단계의 출력이 다음 단계의 입력으로 사용됩니다:
        Analysis → Design → Development → Implementation → Evaluation
        """
        import json

        if self.debug:
            print("\n" + "="*60)
            print("[Sequential ADDIE Pipeline] 순차적 생성 시작")
            print("="*60)

        # 시나리오 컨텍스트 (모든 단계에서 공통 사용)
        scenario_context = self._build_scenario_context(scenario_input)

        # ============================================================
        # Step 1: Analysis 단계
        # ============================================================
        if self.debug:
            print("\n[Step 1/5] Analysis 단계 생성 중...")

        analysis_prompt = f"""## 시나리오 정보
{scenario_context}

## 학습자 프로필
{learner_profile.skill_tree.to_prompt_context()}

위 시나리오에 대한 Analysis(분석) 단계를 수행하세요."""

        analysis_response = self.llm.invoke([
            SystemMessage(content=ANALYSIS_PROMPT),
            HumanMessage(content=analysis_prompt),
        ])
        analysis_data = self._parse_json_response(analysis_response.content)

        if self.debug:
            print(f"  → Analysis 완료: characteristics={len(analysis_data.get('learner_analysis', {}).get('characteristics', []))}개")

        # ============================================================
        # Step 2: Design 단계 (Analysis 결과 입력)
        # ============================================================
        if self.debug:
            print("\n[Step 2/5] Design 단계 생성 중...")

        design_prompt = f"""## 시나리오 정보
{scenario_context}

## 이전 단계 결과: Analysis
```json
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}
```

위 Analysis 결과를 바탕으로 Design(설계) 단계를 수행하세요."""

        design_response = self.llm.invoke([
            SystemMessage(content=DESIGN_PROMPT),
            HumanMessage(content=design_prompt),
        ])
        design_data = self._parse_json_response(design_response.content)

        if self.debug:
            print(f"  → Design 완료: objectives={len(design_data.get('learning_objectives', []))}개, events={len(design_data.get('instructional_strategy', {}).get('sequence', []))}개")

        # ============================================================
        # Step 3: Development 단계 (Analysis + Design 결과 입력)
        # ============================================================
        if self.debug:
            print("\n[Step 3/5] Development 단계 생성 중...")

        development_prompt = f"""## 시나리오 정보
{scenario_context}

## 이전 단계 결과: Analysis
```json
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}
```

## 이전 단계 결과: Design
```json
{json.dumps(design_data, ensure_ascii=False, indent=2)}
```

위 결과를 바탕으로 Development(개발) 단계를 수행하세요.
Design의 learning_objectives ID를 modules의 objectives에 연결하세요."""

        development_response = self.llm.invoke([
            SystemMessage(content=DEVELOPMENT_PROMPT),
            HumanMessage(content=development_prompt),
        ])
        development_data = self._parse_json_response(development_response.content)

        if self.debug:
            print(f"  → Development 완료: modules={len(development_data.get('lesson_plan', {}).get('modules', []))}개, materials={len(development_data.get('materials', []))}개")

        # ============================================================
        # Step 4: Implementation 단계 (이전 단계 결과 입력)
        # ============================================================
        if self.debug:
            print("\n[Step 4/5] Implementation 단계 생성 중...")

        implementation_prompt = f"""## 시나리오 정보
{scenario_context}

## 이전 단계 결과 요약

### Analysis
- 대상: {analysis_data.get('learner_analysis', {}).get('target_audience', '')}
- 환경: {analysis_data.get('context_analysis', {}).get('environment', '')}
- 시간: {analysis_data.get('context_analysis', {}).get('duration', '')}

### Design
- 학습 목표: {len(design_data.get('learning_objectives', []))}개
- 교수 전략: {design_data.get('instructional_strategy', {}).get('model', '')}

### Development - Lesson Plan
```json
{json.dumps(development_data.get('lesson_plan', {}), ensure_ascii=False, indent=2)}
```

위 결과를 바탕으로 Implementation(실행) 단계를 수행하세요.

⚠️ **중요**: facilitator_guide와 learner_guide는 반드시 상세하게 작성하세요!
- facilitator_guide: 200자 이상, 단계별 번호와 시간 배분 포함
- learner_guide: 150자 이상, 학습 전/중/후 구분"""

        implementation_response = self.llm.invoke([
            SystemMessage(content=IMPLEMENTATION_PROMPT),
            HumanMessage(content=implementation_prompt),
        ])
        implementation_data = self._parse_json_response(implementation_response.content)

        if self.debug:
            fg_len = len(implementation_data.get('facilitator_guide', ''))
            lg_len = len(implementation_data.get('learner_guide', ''))
            print(f"  → Implementation 완료: facilitator_guide={fg_len}자, learner_guide={lg_len}자")

        # ============================================================
        # Step 5: Evaluation 단계 (이전 단계 결과 입력)
        # ============================================================
        if self.debug:
            print("\n[Step 5/5] Evaluation 단계 생성 중...")

        evaluation_prompt = f"""## 시나리오 정보
{scenario_context}

## 이전 단계 결과: Design - Learning Objectives
```json
{json.dumps(design_data.get('learning_objectives', []), ensure_ascii=False, indent=2)}
```

위 학습 목표에 맞춰 Evaluation(평가) 단계를 수행하세요.
각 quiz_item의 objective_id가 위 learning_objectives의 id와 연결되어야 합니다."""

        evaluation_response = self.llm.invoke([
            SystemMessage(content=EVALUATION_PROMPT),
            HumanMessage(content=evaluation_prompt),
        ])
        evaluation_data = self._parse_json_response(evaluation_response.content)

        if self.debug:
            print(f"  → Evaluation 완료: quiz_items={len(evaluation_data.get('quiz_items', []))}개")
            print("\n" + "="*60)
            print("[Sequential ADDIE Pipeline] 생성 완료!")
            print("="*60 + "\n")

        # ============================================================
        # 최종 ADDIEOutput 조립
        # ============================================================
        return self._assemble_addie_output(
            scenario_input=scenario_input,
            analysis_data=analysis_data,
            design_data=design_data,
            development_data=development_data,
            implementation_data=implementation_data,
            evaluation_data=evaluation_data,
        )

    def _parse_json_response(self, response_text: str) -> dict:
        """LLM 응답에서 JSON 파싱"""
        import json
        import re

        # JSON 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else response_text

        try:
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 빈 딕셔너리 반환
            return {}

    def _assemble_addie_output(
        self,
        scenario_input: ScenarioInput,
        analysis_data: dict,
        design_data: dict,
        development_data: dict,
        implementation_data: dict,
        evaluation_data: dict,
    ) -> ADDIEOutput:
        """각 단계 데이터를 ADDIEOutput으로 조립"""

        # Analysis
        la_data = analysis_data.get("learner_analysis", {})
        ca_data = analysis_data.get("context_analysis", {})
        ta_data = analysis_data.get("task_analysis", {})
        na_data = analysis_data.get("needs_analysis", {})

        # target_audience 추론
        target_audience = la_data.get("target_audience") or scenario_input.context.target_audience
        if not target_audience:
            parts_audience = []
            if scenario_input.context.learner_age:
                parts_audience.append(scenario_input.context.learner_age)
            if scenario_input.context.learner_role:
                parts_audience.append(scenario_input.context.learner_role)
            target_audience = " ".join(parts_audience) if parts_audience else "학습자"

        # NeedsAnalysis (Item 1-4)
        needs_analysis = None
        if na_data:
            needs_analysis = NeedsAnalysis(
                problem_definition=na_data.get("problem_definition"),
                gap_analysis=na_data.get("gap_analysis"),
                performance_analysis=na_data.get("performance_analysis"),
                needs_prioritization=na_data.get("needs_prioritization"),
            )

        analysis = Analysis(
            learner_analysis=LearnerAnalysis(
                target_audience=target_audience,
                characteristics=la_data.get("characteristics", []),
                prior_knowledge=la_data.get("prior_knowledge", scenario_input.context.prior_knowledge),
                learning_preferences=la_data.get("learning_preferences", []),
                motivation=la_data.get("motivation"),
                challenges=la_data.get("challenges", []),
            ),
            context_analysis=ContextAnalysis(
                environment=ca_data.get("environment", scenario_input.context.learning_environment or "미정"),
                duration=ca_data.get("duration", scenario_input.context.duration or "미정"),
                constraints=ca_data.get("constraints", []),
                resources=ca_data.get("resources", []),
                technical_requirements=ca_data.get("technical_requirements", []),
                # Item 6: 물리/조직/기술 환경 상세
                physical_environment=ca_data.get("physical_environment"),
                organizational_environment=ca_data.get("organizational_environment"),
                technology_environment=ca_data.get("technology_environment"),
            ),
            task_analysis=TaskAnalysis(
                main_topics=ta_data.get("main_topics", scenario_input.learning_goals),
                subtopics=ta_data.get("subtopics", []),
                prerequisites=ta_data.get("prerequisites", []),
                # Item 7-10: 과제 및 목표분석
                initial_learning_objectives=ta_data.get("initial_learning_objectives"),
                sub_skills=ta_data.get("sub_skills", []),
                entry_behaviors=ta_data.get("entry_behaviors"),
                task_analysis_review=ta_data.get("task_analysis_review"),
            ),
            # Item 1-4: 요구분석
            needs_analysis=needs_analysis,
        )

        # Design
        objectives = []
        for i, obj in enumerate(design_data.get("learning_objectives", [])):
            objectives.append(LearningObjective(
                id=obj.get("id", f"OBJ-{i+1:02d}"),
                level=obj.get("level", "이해"),
                statement=obj.get("statement", ""),
                bloom_verb=obj.get("bloom_verb", "설명하다"),
                measurable=obj.get("measurable", True),
            ))

        strategy_data = design_data.get("instructional_strategy", {})
        events = []
        for event in strategy_data.get("sequence", []):
            # resources가 문자열인 경우 리스트로 변환
            resources_val = event.get("resources", [])
            if isinstance(resources_val, str):
                resources_val = [resources_val]
            elif not isinstance(resources_val, list):
                resources_val = []
            else:
                # 리스트 내 None 값 필터링
                resources_val = [r for r in resources_val if r is not None and isinstance(r, str)]

            events.append(InstructionalEvent(
                event=event.get("event", ""),
                activity=event.get("activity", ""),
                duration=event.get("duration"),
                resources=resources_val,
            ))

        assessment_data = design_data.get("assessment_plan", {})
        prototype_data = design_data.get("prototype_design", {})

        # PrototypeDesign (Item 18)
        prototype_design = None
        if prototype_data:
            prototype_design = PrototypeDesign(
                storyboard=prototype_data.get("storyboard"),
                screen_flow=prototype_data.get("screen_flow", []),
                navigation_structure=prototype_data.get("navigation_structure"),
            )

        design = Design(
            learning_objectives=objectives,
            assessment_plan=AssessmentPlan(
                diagnostic=assessment_data.get("diagnostic", []),
                formative=assessment_data.get("formative", []),
                summative=assessment_data.get("summative", []),
            ),
            instructional_strategy=InstructionalStrategy(
                model=strategy_data.get("model", "Gagné's 9 Events"),
                sequence=events,
                methods=strategy_data.get("methods", []),
                # Item 14-17: 교수전략 확장
                instructional_strategies=strategy_data.get("instructional_strategies"),
                non_instructional_strategies=strategy_data.get("non_instructional_strategies"),
                media_selection=strategy_data.get("media_selection", []),
            ),
            # Item 18: 프로토타입 구조 설계
            prototype_design=prototype_design,
        )

        # Development
        lesson_data = development_data.get("lesson_plan", {})
        modules = []
        for mod in lesson_data.get("modules", []):
            activities = []
            for act in mod.get("activities", []):
                # resources 타입 변환 및 None 필터링
                act_resources = act.get("resources", [])
                if isinstance(act_resources, str):
                    act_resources = [act_resources]
                elif isinstance(act_resources, list):
                    act_resources = [r for r in act_resources if r is not None and isinstance(r, str)]
                else:
                    act_resources = []

                activities.append(Activity(
                    time=act.get("time", ""),
                    activity=act.get("activity", ""),
                    description=act.get("description"),
                    resources=act_resources,
                ))
            modules.append(Module(
                title=mod.get("title", ""),
                duration=mod.get("duration", ""),
                objectives=mod.get("objectives", []),
                activities=activities,
            ))

        materials = []
        for mat in development_data.get("materials", []):
            slide_contents_data = mat.get("slide_contents", [])
            slide_contents = None
            if slide_contents_data:
                slide_contents = [
                    SlideContent(
                        slide_number=sc.get("slide_number", i + 1),
                        title=sc.get("title", ""),
                        bullet_points=sc.get("bullet_points", []),
                        speaker_notes=sc.get("speaker_notes"),
                    )
                    for i, sc in enumerate(slide_contents_data)
                ]

            # slides/pages가 숫자 문자열인 경우 정수로 변환
            slides_val = mat.get("slides")
            if isinstance(slides_val, str):
                try:
                    slides_val = int(slides_val)
                except ValueError:
                    slides_val = None

            pages_val = mat.get("pages")
            if isinstance(pages_val, str):
                try:
                    pages_val = int(pages_val)
                except ValueError:
                    pages_val = None

            materials.append(Material(
                type=mat.get("type", ""),
                title=mat.get("title", ""),
                description=mat.get("description"),
                slides=slides_val,
                duration=mat.get("duration"),
                pages=pages_val,
                slide_contents=slide_contents,
            ))

        development = Development(
            lesson_plan=LessonPlan(
                total_duration=lesson_data.get("total_duration", scenario_input.context.duration or "미정"),
                modules=modules,
            ),
            materials=materials,
            # Item 20-23: 개발 단계 확장
            facilitator_manual=development_data.get("facilitator_manual"),
            operator_manual=development_data.get("operator_manual"),
            assessment_tools=development_data.get("assessment_tools", []),
            expert_review_plan=development_data.get("expert_review_plan"),
        )

        # Implementation
        implementation = Implementation(
            delivery_method=implementation_data.get("delivery_method", "대면 교육"),
            facilitator_guide=implementation_data.get("facilitator_guide"),
            learner_guide=implementation_data.get("learner_guide"),
            technical_requirements=implementation_data.get("technical_requirements", []),
            support_plan=implementation_data.get("support_plan"),
            # Item 24-27: 실행 단계 확장
            orientation_plan=implementation_data.get("orientation_plan"),
            system_check_plan=implementation_data.get("system_check_plan"),
            pilot_execution_plan=implementation_data.get("pilot_execution_plan"),
            monitoring_plan=implementation_data.get("monitoring_plan"),
        )

        # Evaluation
        quiz_items = []
        for i, item in enumerate(evaluation_data.get("quiz_items", [])):
            # answer가 리스트인 경우 쉼표로 연결 (다답형 문항 처리)
            answer_val = item.get("answer", "")
            if isinstance(answer_val, list):
                answer_val = ", ".join(str(a) for a in answer_val)
            elif not isinstance(answer_val, str):
                answer_val = str(answer_val) if answer_val is not None else ""

            # options가 dict 리스트인 경우 문자열 리스트로 변환
            options_val = item.get("options", [])
            if isinstance(options_val, list):
                options_val = [
                    str(opt.get("text", opt.get("label", str(opt)))) if isinstance(opt, dict) else str(opt)
                    for opt in options_val if opt is not None
                ]
            else:
                options_val = []

            quiz_items.append(QuizItem(
                id=item.get("id", f"Q-{i+1:02d}"),
                question=item.get("question", ""),
                type=item.get("type", "multiple_choice"),
                options=options_val,
                answer=answer_val,
                explanation=item.get("explanation"),
                objective_id=item.get("objective_id"),
                difficulty=item.get("difficulty"),
            ))

        rubric_data = evaluation_data.get("rubric")
        rubric = None
        if rubric_data:
            rubric = Rubric(
                criteria=rubric_data.get("criteria", []),
                levels=rubric_data.get("levels", {}),
            )

        evaluation = Evaluation(
            quiz_items=quiz_items,
            rubric=rubric,
            feedback_plan=evaluation_data.get("feedback_plan"),
            # Item 28-33: 형성평가/총괄평가/프로그램 개선 필드
            pilot_data_collection=evaluation_data.get("pilot_data_collection"),
            formative_improvement=evaluation_data.get("formative_improvement"),
            summative_evaluation_plan=evaluation_data.get("summative_evaluation_plan"),
            adoption_decision_criteria=evaluation_data.get("adoption_decision_criteria"),
            program_improvement=evaluation_data.get("program_improvement"),
        )

        return ADDIEOutput(
            analysis=analysis,
            design=design,
            development=development,
            implementation=implementation,
            evaluation=evaluation,
        )

    def _validate_minimum_requirements(self, addie_output: ADDIEOutput) -> tuple[int, list[str]]:
        """최소 요구사항 검증 (0-100 점수 반환)"""
        issues = []
        score = 100

        # Analysis 검증
        la = addie_output.analysis.learner_analysis
        if len(la.characteristics) < 5:
            issues.append(f"characteristics: {len(la.characteristics)}/5")
            score -= 10
        if len(la.learning_preferences) < 4:
            issues.append(f"learning_preferences: {len(la.learning_preferences)}/4")
            score -= 5
        if len(la.challenges) < 3:
            issues.append(f"challenges: {len(la.challenges)}/3")
            score -= 5

        # Design 검증
        d = addie_output.design
        if len(d.learning_objectives) < 5:
            issues.append(f"learning_objectives: {len(d.learning_objectives)}/5")
            score -= 20  # 학습 목표는 중요
        if len(d.instructional_strategy.sequence) < 9:
            issues.append(f"instructional_strategy.sequence: {len(d.instructional_strategy.sequence)}/9")
            score -= 15

        # Development 검증
        dev = addie_output.development
        if len(dev.lesson_plan.modules) < 3:
            issues.append(f"modules: {len(dev.lesson_plan.modules)}/3")
            score -= 10
        if len(dev.materials) < 5:
            issues.append(f"materials: {len(dev.materials)}/5")
            score -= 5

        # Evaluation 검증
        ev = addie_output.evaluation
        if len(ev.quiz_items) < 10:
            issues.append(f"quiz_items: {len(ev.quiz_items)}/10")
            score -= 10

        # Implementation 검증 (가이드 길이)
        impl = addie_output.implementation
        fg_len = len(str(impl.facilitator_guide or ""))
        lg_len = len(str(impl.learner_guide or ""))
        if fg_len < 200:
            issues.append(f"facilitator_guide: {fg_len}/200자")
            score -= 15  # 중요한 항목
        if lg_len < 100:
            issues.append(f"learner_guide: {lg_len}/100자")
            score -= 10

        return max(0, score), issues

    def _build_generation_prompt(
        self,
        scenario_input: ScenarioInput,
        learner_profile: LearnerProfile,
    ) -> str:
        """생성 프롬프트 구성"""
        parts = [
            "다음 시나리오에 맞는 ADDIE 교수설계 산출물을 생성해주세요.\n",
            "## 시나리오 정보",
            f"**제목:** {scenario_input.title}",
            f"**시나리오 ID:** {scenario_input.scenario_id}",
            f"\n**학습 맥락:**",
            f"- 대상: {scenario_input.context.target_audience}",
            f"- 시간: {scenario_input.context.duration}",
            f"- 환경: {scenario_input.context.learning_environment}",
        ]

        if scenario_input.context.prior_knowledge:
            parts.append(f"- 사전지식: {scenario_input.context.prior_knowledge}")

        if scenario_input.context.class_size:
            parts.append(f"- 학습자 수: {scenario_input.context.class_size}명")

        parts.append(f"\n**학습 목표:**")
        for goal in scenario_input.learning_goals:
            parts.append(f"- {goal}")

        if scenario_input.constraints:
            parts.append("\n**제약 조건:**")
            if scenario_input.constraints.budget:
                parts.append(f"- 예산: {scenario_input.constraints.budget}")
            if scenario_input.constraints.resources:
                parts.append(f"- 사용 가능 자원: {', '.join(scenario_input.constraints.resources)}")
            if scenario_input.constraints.accessibility:
                parts.append(f"- 접근성: {', '.join(scenario_input.constraints.accessibility)}")

        # 학습자 프로필
        parts.append("\n" + learner_profile.skill_tree.to_prompt_context())

        parts.append("\n위 정보를 바탕으로 완전한 ADDIE 산출물을 생성해주세요.")

        return "\n".join(parts)

    def _parse_addie_response(
        self,
        response_text: str,
        scenario_input: ScenarioInput
    ) -> ADDIEOutput:
        """LLM 응답을 ADDIEOutput으로 파싱"""
        import json
        import re

        # JSON 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            return self._dict_to_addie_output(data, scenario_input)

        # JSON 블록 없이 직접 파싱 시도
        try:
            data = json.loads(response_text)
            return self._dict_to_addie_output(data, scenario_input)
        except json.JSONDecodeError:
            return self._create_default_output(scenario_input)

    def _dict_to_addie_output(
        self,
        data: dict,
        scenario_input: ScenarioInput
    ) -> ADDIEOutput:
        """딕셔너리를 ADDIEOutput으로 변환"""
        # Analysis
        analysis_data = data.get("analysis", {})
        analysis = Analysis(
            learner_analysis=LearnerAnalysis(
                target_audience=scenario_input.context.target_audience,
                characteristics=analysis_data.get("learner_analysis", {}).get("characteristics", []),
                prior_knowledge=scenario_input.context.prior_knowledge,
                learning_preferences=analysis_data.get("learner_analysis", {}).get("learning_preferences", []),
                motivation=analysis_data.get("learner_analysis", {}).get("motivation"),
                challenges=analysis_data.get("learner_analysis", {}).get("challenges", []),
            ),
            context_analysis=ContextAnalysis(
                environment=scenario_input.context.learning_environment,
                duration=scenario_input.context.duration,
                constraints=analysis_data.get("context_analysis", {}).get("constraints", []),
                resources=analysis_data.get("context_analysis", {}).get("resources", []),
                technical_requirements=analysis_data.get("context_analysis", {}).get("technical_requirements", []),
            ),
            task_analysis=TaskAnalysis(
                main_topics=analysis_data.get("task_analysis", {}).get("main_topics", scenario_input.learning_goals),
                subtopics=analysis_data.get("task_analysis", {}).get("subtopics", []),
                prerequisites=analysis_data.get("task_analysis", {}).get("prerequisites", []),
            ),
        )

        # Design
        design_data = data.get("design", {})
        objectives = []
        for i, obj in enumerate(design_data.get("learning_objectives", [])):
            objectives.append(LearningObjective(
                id=f"OBJ-{i+1:02d}",
                level=obj.get("level", "이해"),
                statement=obj.get("statement", ""),
                bloom_verb=obj.get("bloom_verb", "설명하다"),
                measurable=obj.get("measurable", True),
            ))

        strategy_data = design_data.get("instructional_strategy", {})
        events = []
        for event in strategy_data.get("sequence", []):
            # resources가 문자열인 경우 리스트로 변환
            resources_val = event.get("resources", [])
            if isinstance(resources_val, str):
                resources_val = [resources_val]
            elif not isinstance(resources_val, list):
                resources_val = []
            else:
                # 리스트 내 None 값 필터링
                resources_val = [r for r in resources_val if r is not None and isinstance(r, str)]

            events.append(InstructionalEvent(
                event=event.get("event", ""),
                activity=event.get("activity", ""),
                duration=event.get("duration"),
                resources=resources_val,
            ))

        design = Design(
            learning_objectives=objectives,
            assessment_plan=AssessmentPlan(
                formative=design_data.get("assessment_plan", {}).get("formative", []),
                summative=design_data.get("assessment_plan", {}).get("summative", []),
                diagnostic=design_data.get("assessment_plan", {}).get("diagnostic", []),
            ),
            instructional_strategy=InstructionalStrategy(
                model=strategy_data.get("model", "Gagné's 9 Events"),
                sequence=events,
                methods=strategy_data.get("methods", []),
            ),
        )

        # Development
        dev_data = data.get("development", {})
        modules = []
        for mod in dev_data.get("lesson_plan", {}).get("modules", []):
            activities = []
            for act in mod.get("activities", []):
                # resources 타입 변환 및 None 필터링
                act_resources = act.get("resources", [])
                if isinstance(act_resources, str):
                    act_resources = [act_resources]
                elif isinstance(act_resources, list):
                    act_resources = [r for r in act_resources if r is not None and isinstance(r, str)]
                else:
                    act_resources = []

                activities.append(Activity(
                    time=act.get("time", ""),
                    activity=act.get("activity", ""),
                    description=act.get("description"),
                    resources=act_resources,
                ))
            modules.append(Module(
                title=mod.get("title", ""),
                duration=mod.get("duration", ""),
                objectives=mod.get("objectives", []),
                activities=activities,
            ))

        materials = []
        for mat in dev_data.get("materials", []):
            # slide_contents 파싱
            slide_contents_data = mat.get("slide_contents", [])
            slide_contents = None
            if slide_contents_data:
                slide_contents = [
                    SlideContent(
                        slide_number=sc.get("slide_number", i + 1),
                        title=sc.get("title", ""),
                        bullet_points=sc.get("bullet_points", []),
                        speaker_notes=sc.get("speaker_notes"),
                        visual_suggestion=sc.get("visual_suggestion"),
                    )
                    for i, sc in enumerate(slide_contents_data)
                ]

            # Fallback: 프레젠테이션인데 slide_contents가 없으면 자동 생성
            mat_type = mat.get("type", "").lower()
            if not slide_contents and ("프레젠테이션" in mat_type or "슬라이드" in mat_type or "presentation" in mat_type):
                slide_contents = self._generate_fallback_slides(modules, scenario_input, objectives)

            # slides/pages가 숫자 문자열인 경우 정수로 변환
            slides_val = mat.get("slides")
            if slides_val is None and slide_contents:
                slides_val = len(slide_contents)
            elif isinstance(slides_val, str):
                try:
                    slides_val = int(slides_val)
                except ValueError:
                    slides_val = len(slide_contents) if slide_contents else None

            pages_val = mat.get("pages")
            if isinstance(pages_val, str):
                try:
                    pages_val = int(pages_val)
                except ValueError:
                    pages_val = None

            materials.append(Material(
                type=mat.get("type", ""),
                title=mat.get("title", ""),
                description=mat.get("description"),
                slides=slides_val,
                duration=mat.get("duration"),
                pages=pages_val,
                slide_contents=slide_contents,
            ))

        # Fallback: materials 처리 후 slide_contents가 있는 material이 없으면 자동 생성
        has_slide_contents = any(mat.slide_contents for mat in materials)
        if not has_slide_contents and modules:
            fallback_slides = self._generate_fallback_slides(modules, scenario_input, objectives)
            if fallback_slides:
                materials.append(Material(
                    type="프레젠테이션",
                    title="교육 슬라이드",
                    description="모듈 정보 기반 자동 생성 슬라이드",
                    slides=len(fallback_slides),
                    slide_contents=fallback_slides,
                ))

        development = Development(
            lesson_plan=LessonPlan(
                total_duration=scenario_input.context.duration,
                modules=modules,
            ),
            materials=materials,
        )

        # Implementation
        impl_data = data.get("implementation", {})
        implementation = Implementation(
            delivery_method=impl_data.get("delivery_method", "대면 교육"),
            facilitator_guide=impl_data.get("facilitator_guide"),
            learner_guide=impl_data.get("learner_guide"),
            technical_requirements=impl_data.get("technical_requirements", []),
            support_plan=impl_data.get("support_plan"),
        )

        # Evaluation
        eval_data = data.get("evaluation", {})
        quiz_items = []
        for i, item in enumerate(eval_data.get("quiz_items", [])):
            # answer가 리스트인 경우 쉼표로 연결 (다답형 문항 처리)
            answer_val = item.get("answer", "")
            if isinstance(answer_val, list):
                answer_val = ", ".join(str(a) for a in answer_val)
            elif not isinstance(answer_val, str):
                answer_val = str(answer_val) if answer_val is not None else ""

            # options가 dict 리스트인 경우 문자열 리스트로 변환
            options_val = item.get("options", [])
            if isinstance(options_val, list):
                options_val = [
                    str(opt.get("text", opt.get("label", str(opt)))) if isinstance(opt, dict) else str(opt)
                    for opt in options_val if opt is not None
                ]
            else:
                options_val = []

            quiz_items.append(QuizItem(
                id=f"Q-{i+1:02d}",
                question=item.get("question", ""),
                type=item.get("type", "multiple_choice"),
                options=options_val,
                answer=answer_val,
                explanation=item.get("explanation"),
                objective_id=item.get("objective_id"),
                difficulty=item.get("difficulty"),
            ))

        rubric_data = eval_data.get("rubric")
        rubric = None
        if rubric_data:
            rubric = Rubric(
                criteria=rubric_data.get("criteria", []),
                levels=rubric_data.get("levels", {}),
            )

        evaluation = Evaluation(
            quiz_items=quiz_items,
            rubric=rubric,
            feedback_plan=eval_data.get("feedback_plan"),
            # Item 28-33: 형성평가/총괄평가/프로그램 개선 필드
            pilot_data_collection=eval_data.get("pilot_data_collection"),
            formative_improvement=eval_data.get("formative_improvement"),
            summative_evaluation_plan=eval_data.get("summative_evaluation_plan"),
            adoption_decision_criteria=eval_data.get("adoption_decision_criteria"),
            program_improvement=eval_data.get("program_improvement"),
        )

        return ADDIEOutput(
            analysis=analysis,
            design=design,
            development=development,
            implementation=implementation,
            evaluation=evaluation,
        )

    def _generate_fallback_slides(
        self,
        modules: list[Module],
        scenario_input: ScenarioInput,
        learning_objectives: list[LearningObjective] = None,
    ) -> list[SlideContent]:
        """모듈 정보를 기반으로 폴백 슬라이드 콘텐츠 생성"""
        slide_contents = []
        slide_num = 1

        # objective ID → statement 매핑 생성
        obj_id_to_statement = {}
        if learning_objectives:
            for obj in learning_objectives:
                obj_id_to_statement[obj.id] = obj.statement

        def resolve_objective(obj_ref: str) -> str:
            """objective ID를 실제 statement로 변환, 없으면 원본 반환"""
            if obj_ref in obj_id_to_statement:
                return obj_id_to_statement[obj_ref]
            # OBJ-xx 패턴이면 매핑에서 찾기
            if obj_ref.startswith("OBJ-"):
                return obj_id_to_statement.get(obj_ref, obj_ref)
            return obj_ref

        # 도입 슬라이드
        slide_contents.append(SlideContent(
            slide_number=slide_num,
            title="교육 소개",
            bullet_points=["환영 인사", "교육 목표", "일정 안내"],
            speaker_notes="참가자들을 환영하며 교육 목표를 명확히 전달합니다.",
        ))
        slide_num += 1

        # 학습 목표 슬라이드
        if scenario_input.learning_goals:
            slide_contents.append(SlideContent(
                slide_number=slide_num,
                title="학습 목표",
                bullet_points=scenario_input.learning_goals[:5],
                speaker_notes="오늘 학습을 통해 달성할 목표들을 설명합니다.",
            ))
            slide_num += 1

        # 모듈별 슬라이드 생성
        for module in modules:
            module_title = module.title if module.title else "학습 모듈"

            # 모듈 시작 슬라이드 - objective ID를 실제 statement로 변환
            raw_objectives = module.objectives[:3] if module.objectives else []
            resolved_objectives = [resolve_objective(obj) for obj in raw_objectives] if raw_objectives else ["학습 목표", "주요 내용"]

            bullet_points = resolved_objectives + [f"예상 소요 시간: {module.duration}"] if module.duration else resolved_objectives
            slide_contents.append(SlideContent(
                slide_number=slide_num,
                title=module_title,
                bullet_points=bullet_points,
                speaker_notes=f"{module_title}의 학습 목표와 개요를 설명합니다.",
            ))
            slide_num += 1

            # 활동별 슬라이드 (최대 3개)
            activities = module.activities if module.activities else []
            for activity in activities[:3]:
                activity_name = activity.activity if activity.activity else "학습 활동"
                description = activity.description if activity.description else ""
                bullet_points = [description] if description else ["활동 설명"]

                # 자원 정보 추가
                if activity.resources:
                    bullet_points.extend([f"자원: {r}" for r in activity.resources[:2]])

                slide_contents.append(SlideContent(
                    slide_number=slide_num,
                    title=activity_name,
                    bullet_points=bullet_points,
                    speaker_notes=f"{activity_name} 진행 방법을 안내합니다.",
                ))
                slide_num += 1

        # 마무리 슬라이드
        slide_contents.append(SlideContent(
            slide_number=slide_num,
            title="정리 및 Q&A",
            bullet_points=["오늘 학습 내용 요약", "핵심 포인트 정리", "질의응답"],
            speaker_notes="핵심 내용을 요약하고 질문을 받습니다.",
        ))

        return slide_contents

    def _create_default_output(self, scenario_input: ScenarioInput) -> ADDIEOutput:
        """기본 ADDIE 산출물 생성 (파싱 실패 시)"""
        return ADDIEOutput(
            analysis=Analysis(
                learner_analysis=LearnerAnalysis(
                    target_audience=scenario_input.context.target_audience,
                    characteristics=[],
                    prior_knowledge=scenario_input.context.prior_knowledge,
                ),
                context_analysis=ContextAnalysis(
                    environment=scenario_input.context.learning_environment,
                    duration=scenario_input.context.duration,
                ),
                task_analysis=TaskAnalysis(
                    main_topics=scenario_input.learning_goals,
                ),
            ),
            design=Design(
                learning_objectives=[
                    LearningObjective(
                        id="OBJ-01",
                        level="이해",
                        statement=scenario_input.learning_goals[0] if scenario_input.learning_goals else "",
                        bloom_verb="설명하다",
                    )
                ],
                assessment_plan=AssessmentPlan(),
                instructional_strategy=InstructionalStrategy(),
            ),
            development=Development(
                lesson_plan=LessonPlan(
                    total_duration=scenario_input.context.duration,
                ),
            ),
            implementation=Implementation(
                delivery_method="대면 교육",
            ),
            evaluation=Evaluation(),
        )
