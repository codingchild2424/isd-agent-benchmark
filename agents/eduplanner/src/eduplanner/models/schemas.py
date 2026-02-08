"""EduPlanner 입출력 스키마 정의"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class ContextInfo(BaseModel):
    """학습 맥락 정보"""
    # 기존 필드 (optional로 변경)
    target_audience: Optional[str] = Field(None, description="학습 대상자")
    prior_knowledge: Optional[str] = Field(None, description="사전 지식 수준")
    duration: Optional[str] = Field(None, description="학습 시간")
    learning_environment: Optional[str] = Field(None, description="학습 환경")
    class_size: Optional[str] = Field(None, description="학습자 수 (문자열)")
    additional_context: Optional[str] = Field(None, description="추가 맥락")
    # IDLD 시나리오 추가 필드
    institution_type: Optional[str] = Field(None, description="기관 유형")
    learner_age: Optional[str] = Field(None, description="학습자 연령")
    learner_education: Optional[str] = Field(None, description="학습자 학력")
    learner_role: Optional[str] = Field(None, description="학습자 역할")
    domain_expertise: Optional[str] = Field(None, description="도메인 전문성")


class Constraints(BaseModel):
    """제약 조건"""
    budget: Optional[str] = Field(None, description="예산 수준")
    resources: Optional[list[str]] = Field(None, description="사용 가능한 자료")
    accessibility: Optional[Any] = Field(None, description="접근성 요구사항")
    language: str = Field(default="ko", description="콘텐츠 언어")
    # IDLD 시나리오 추가 필드
    tech_requirements: Optional[str] = Field(None, description="기술 요구사항")
    assessment_type: Optional[str] = Field(None, description="평가 유형")


class ScenarioInput(BaseModel):
    """교수설계 시나리오 입력"""
    scenario_id: str = Field(..., description="시나리오 고유 식별자")
    variant_type: Optional[str] = Field(None, description="시나리오 유형")
    title: str = Field(..., description="시나리오 제목")
    context: ContextInfo = Field(..., description="학습 맥락")
    learning_goals: list[str] = Field(..., description="학습 목표")
    constraints: Optional[Constraints] = Field(None, description="제약 조건")
    difficulty: Optional[str] = Field(None, description="난이도")
    domain: Optional[str] = Field(None, description="교육 도메인")


class LearnerAnalysis(BaseModel):
    """학습자 분석 (Item 5: 학습자 분석)"""
    target_audience: str
    characteristics: list[str] = Field(default_factory=list)
    prior_knowledge: Optional[str] = None
    learning_preferences: list[str] = Field(default_factory=list)
    motivation: Optional[str] = None
    challenges: list[str] = Field(default_factory=list)


class ContextAnalysis(BaseModel):
    """환경 분석 (Item 6: 환경 분석 - 물리/조직/기술 환경)"""
    environment: str
    duration: str
    constraints: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    technical_requirements: list[str] = Field(default_factory=list)
    # Item 6 확장: 물리적, 조직적, 기술적 환경 상세
    physical_environment: Optional[str] = Field(default=None, description="물리적 환경 분석")
    organizational_environment: Optional[str] = Field(default=None, description="조직적 환경 분석")
    technology_environment: Optional[str] = Field(default=None, description="기술적 환경 분석")


class TaskAnalysis(BaseModel):
    """과제 분석 (Item 7-10: 과제 및 목표분석)"""
    main_topics: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    # Item 7: 초기 학습목표 분석
    initial_learning_objectives: Optional[str] = Field(default=None, description="초기 학습목표 분석")
    # Item 8: 하위 기능 분석
    sub_skills: list[str] = Field(default_factory=list, description="하위 기능/스킬 분석")
    # Item 9: 출발점 행동 분석
    entry_behaviors: Optional[str] = Field(default=None, description="출발점 행동 분석")
    # Item 10: 과제분석 결과 검토·정리
    task_analysis_review: Optional[str] = Field(default=None, description="과제분석 결과 검토 및 정리")


class NeedsAnalysis(BaseModel):
    """요구분석 (Item 1-4)"""
    # Item 1: 문제 확인 및 정의
    problem_definition: Optional[str] = Field(default=None, description="문제 확인 및 정의")
    # Item 2: 차이분석 (현재-목표 상태 격차)
    gap_analysis: Optional[str] = Field(default=None, description="현재와 목표 성과 간 차이 분석")
    # Item 3: 수행분석
    performance_analysis: Optional[str] = Field(default=None, description="수행분석 결과")
    # Item 4: 요구 우선순위 결정
    needs_prioritization: Optional[str] = Field(default=None, description="요구 우선순위 결정")


class Analysis(BaseModel):
    """분석 단계 산출물 (Item 1-10)"""
    learner_analysis: LearnerAnalysis
    context_analysis: ContextAnalysis
    task_analysis: TaskAnalysis
    # 요구분석 (Item 1-4) - 신규 추가
    needs_analysis: Optional[NeedsAnalysis] = Field(default=None, description="요구분석 (Item 1-4)")


class LearningObjective(BaseModel):
    """학습 목표"""
    id: str
    level: str = Field(..., description="Bloom's Taxonomy 수준")
    statement: str
    bloom_verb: str
    measurable: bool = True


class AssessmentPlan(BaseModel):
    """평가 계획"""
    formative: list[str] = Field(default_factory=list)
    summative: list[str] = Field(default_factory=list)
    diagnostic: list[str] = Field(default_factory=list)


class InstructionalEvent(BaseModel):
    """교수 사태 (Gagné's 9 Events)"""
    event: str
    activity: str
    duration: Optional[str] = None
    resources: list[str] = Field(default_factory=list)


class InstructionalStrategy(BaseModel):
    """교수 전략 (Item 13-17)"""
    model: str = Field(default="Gagné's 9 Events")
    sequence: list[InstructionalEvent] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    # Item 13: 교수 내용 선정 (기존 methods로 커버)
    # Item 14: 교수적 전략 수립
    instructional_strategies: Optional[str] = Field(default=None, description="교수적 전략 수립")
    # Item 15: 비교수적 전략 수립 (신규)
    non_instructional_strategies: Optional[str] = Field(default=None, description="비교수적 전략 (동기부여, 자기주도학습 촉진 등)")
    # Item 16: 매체 선정과 활용 계획 (신규)
    media_selection: list[str] = Field(default_factory=list, description="매체 선정 및 활용 계획")
    # Item 17: 학습활동 및 시간 구조화 (기존 sequence로 커버)


class PrototypeDesign(BaseModel):
    """프로토타입 구조 설계 (Item 18)"""
    # Item 18: 스토리보드/화면 흐름 설계
    storyboard: Optional[str] = Field(default=None, description="스토리보드 설계")
    screen_flow: list[str] = Field(default_factory=list, description="화면 흐름 설계")
    navigation_structure: Optional[str] = Field(default=None, description="네비게이션 구조")


class Design(BaseModel):
    """설계 단계 산출물 (Item 11-18)"""
    learning_objectives: list[LearningObjective]
    assessment_plan: AssessmentPlan
    instructional_strategy: InstructionalStrategy
    # Item 18: 프로토타입 구조 설계 (신규)
    prototype_design: Optional[PrototypeDesign] = Field(default=None, description="프로토타입 구조 설계 (스토리보드/화면 흐름)")


class Activity(BaseModel):
    """학습 활동"""
    time: str
    activity: str
    description: Optional[str] = None
    resources: list[str] = Field(default_factory=list)


class Module(BaseModel):
    """학습 모듈"""
    title: str
    duration: str
    objectives: list[str] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)


class LessonPlan(BaseModel):
    """레슨 플랜"""
    total_duration: str
    modules: list[Module] = Field(default_factory=list)


class SlideContent(BaseModel):
    """개별 슬라이드 콘텐츠"""
    slide_number: int = Field(..., description="슬라이드 번호")
    title: str = Field(..., description="슬라이드 제목")
    bullet_points: list[str] = Field(default_factory=list, description="핵심 내용 (3-5개)")
    speaker_notes: Optional[str] = Field(default=None, description="발표자 노트")
    visual_suggestion: Optional[str] = Field(default=None, description="권장 시각 자료")


class Material(BaseModel):
    """학습 자료"""
    type: str
    title: str
    description: Optional[str] = None
    slides: Optional[int] = None
    duration: Optional[str] = None
    questions: Optional[int] = None
    pages: Optional[int] = None
    content: Optional[str] = Field(default=None, description="실제 학습 자료 내용 (유인물 텍스트, 슬라이드 개요 등)")
    slide_contents: Optional[list[SlideContent]] = Field(default=None, description="슬라이드별 상세 콘텐츠 (PPT인 경우)")


class Development(BaseModel):
    """개발 단계 산출물 (Item 19-23)"""
    lesson_plan: LessonPlan
    # Item 19: 학습자용 자료 개발 (기존 materials)
    materials: list[Material] = Field(default_factory=list)
    # Item 20: 교수자용 매뉴얼 개발 (신규)
    facilitator_manual: Optional[str] = Field(default=None, description="교수자용 매뉴얼")
    # Item 21: 운영자용 매뉴얼 개발 (신규)
    operator_manual: Optional[str] = Field(default=None, description="운영자용 매뉴얼")
    # Item 22: 평가 도구·문항 개발 (evaluation 단계로 이동, 여기서는 평가 도구 명세)
    assessment_tools: list[str] = Field(default_factory=list, description="평가 도구 명세")
    # Item 23: 전문가 검토 (신규)
    expert_review_plan: Optional[str] = Field(default=None, description="전문가 검토 계획 및 피드백 반영 방안")


class Implementation(BaseModel):
    """실행 단계 산출물 (Item 24-27)"""
    delivery_method: str
    facilitator_guide: Optional[str] = None
    learner_guide: Optional[str] = None
    technical_requirements: list[str] = Field(default_factory=list)
    support_plan: Optional[str] = None
    # Item 24: 교수자·운영자 오리엔테이션 (신규)
    orientation_plan: Optional[str] = Field(default=None, description="교수자·운영자 오리엔테이션 계획")
    # Item 25: 시스템/환경 점검 (신규)
    system_check_plan: Optional[str] = Field(default=None, description="시스템/환경 점검 계획")
    # Item 26: 프로토타입 실행 (신규)
    pilot_execution_plan: Optional[str] = Field(default=None, description="프로토타입/파일럿 실행 계획")
    # Item 27: 운영 모니터링 및 지원 (신규)
    monitoring_plan: Optional[str] = Field(default=None, description="운영 모니터링 및 지원 계획")


class QuizItem(BaseModel):
    """퀴즈 문항"""
    id: str
    question: str
    type: str
    options: list[str] = Field(default_factory=list)
    answer: str
    explanation: Optional[str] = None
    objective_id: Optional[str] = None
    difficulty: Optional[str] = None


class Rubric(BaseModel):
    """평가 루브릭"""
    criteria: list[str] = Field(default_factory=list)
    levels: dict[str, str] = Field(default_factory=dict)


class Evaluation(BaseModel):
    """평가 단계 산출물 (Item 28-33)"""
    quiz_items: list[QuizItem] = Field(default_factory=list)
    rubric: Optional[Rubric] = None
    feedback_plan: Optional[str] = None
    # Item 28: 파일럿/초기 실행 중 자료 수집 (신규)
    pilot_data_collection: Optional[str] = Field(default=None, description="파일럿/초기 실행 중 자료 수집 계획")
    # Item 29: 형성평가 결과 기반 1차 프로그램 개선 (신규)
    formative_improvement: Optional[str] = Field(default=None, description="형성평가 결과 기반 1차 프로그램 개선 계획")
    # Item 30: 총괄 평가 문항 개발 (quiz_items로 커버)
    # Item 31: 총괄평가 시행 및 프로그램 효과 분석 (신규)
    summative_evaluation_plan: Optional[str] = Field(default=None, description="총괄평가 시행 및 프로그램 효과 분석 계획")
    # Item 32: 프로그램 채택 여부 결정 (신규)
    adoption_decision_criteria: Optional[str] = Field(default=None, description="프로그램 채택 여부 결정 기준")
    # Item 33: 프로그램 개선 (신규)
    program_improvement: Optional[str] = Field(default=None, description="프로그램 개선 및 환류 계획")


class ADDIEOutput(BaseModel):
    """ADDIE 5단계 산출물"""
    analysis: Analysis
    design: Design
    development: Development
    implementation: Implementation
    evaluation: Evaluation

    def to_standard_dict(self) -> dict:
        """
        표준 ADDIE 33개 소항목 스키마로 변환 (docs/addie_output_schema.json 준수)

        평가기(evaluator)가 기대하는 필드 경로에 맞게 변환합니다.
        """
        # Analysis 단계 변환
        la = self.analysis.learner_analysis
        ca = self.analysis.context_analysis
        ta = self.analysis.task_analysis
        na = self.analysis.needs_analysis

        analysis_dict = {
            # A1: 요구분석 (소항목 1-4)
            "needs_analysis": {
                # [1] 문제 확인 및 정의
                "problem_definition": (na.problem_definition if na else "") or ta.task_analysis_review or "",
                # [2] 차이분석
                "gap_analysis": [{"description": na.gap_analysis}] if na and na.gap_analysis else [],
                # [3] 수행분석
                "performance_analysis": na.performance_analysis if na else "",
                # [4] 요구 우선순위 결정
                "priority_matrix": {
                    "prioritization": na.needs_prioritization if na else "",
                    "high_priority": ta.main_topics[:2] if ta.main_topics else [],
                },
            },
            # A2: 학습자 및 환경분석 (소항목 5-6)
            "learner_analysis": {
                "target_audience": la.target_audience,
                "characteristics": la.characteristics,
                "prior_knowledge": la.prior_knowledge or "",
                "learning_preferences": la.learning_preferences,
                "motivation": la.motivation or "",
            },
            "context_analysis": {
                "environment": ca.environment,
                "constraints": ca.constraints,
                "resources": ca.resources,
                "technical_requirements": ca.technical_requirements,
            },
            # A3: 과제 및 목표분석 (소항목 7-10)
            "task_analysis": {
                # [7] 초기 학습목표 분석
                "initial_objectives": [ta.initial_learning_objectives] if ta.initial_learning_objectives else ta.main_topics,
                # [8] 하위 기능 분석
                "subtopics": ta.sub_skills if ta.sub_skills else ta.subtopics,
                # [9] 출발점 행동 분석
                "prerequisites": ta.prerequisites,
                # [10] 과제분석 결과 검토·정리
                "review_summary": ta.task_analysis_review or f"주요 주제: {', '.join(ta.main_topics[:3])}",
            },
        }

        # Design 단계 변환
        d = self.design
        ist = d.instructional_strategy
        pd = d.prototype_design

        # 학습 활동 생성
        learning_activities = []
        for event in ist.sequence:
            learning_activities.append({
                "activity_name": event.event,
                "duration": event.duration or "",
                "description": event.activity,
                "materials": event.resources,
            })

        design_dict = {
            # [11] 학습목표 정교화
            "learning_objectives": [
                {
                    "id": obj.id,
                    "level": obj.level,
                    "statement": obj.statement,
                    "bloom_verb": obj.bloom_verb,
                    "measurable": obj.measurable,
                }
                for obj in d.learning_objectives
            ],
            # [12] 평가 계획 수립
            "assessment_plan": {
                "formative": [{"description": f} for f in d.assessment_plan.formative],
                "summative": [{"description": s} for s in d.assessment_plan.summative],
                "assessment_criteria": d.assessment_plan.diagnostic,
            },
            # [13] 교수 내용 선정
            "content_structure": {
                "modules": [m for m in ist.methods],
                "topics": [],
                "sequencing": ist.model,
            },
            # [14] 교수적 전략 수립
            "instructional_strategies": {
                "methods": ist.methods,
                "activities": [e.activity for e in ist.sequence[:5]],
                "rationale": ist.instructional_strategies or f"모델: {ist.model}",
            },
            # [15] 비교수적 전략 수립
            "non_instructional_strategies": {
                "motivation_strategies": [ist.non_instructional_strategies] if ist.non_instructional_strategies else [],
                "self_directed_learning": [],
                "support_strategies": [],
            },
            # [16] 매체 선정과 활용 계획
            "media_selection": {
                "media_types": ist.media_selection,
                "tools": [],
                "utilization_plan": "",
            },
            # [17] 학습활동 및 시간 구조화
            "learning_activities": learning_activities,
            # [18] 스토리보드/화면 흐름 설계
            "storyboard": {
                "screens": pd.screen_flow if pd else [],
                "navigation_flow": pd.navigation_structure if pd else "",
                "interactions": [],
            } if pd else {"screens": [], "navigation_flow": "", "interactions": []},
        }

        # Development 단계 변환
        dev = self.development

        # learner_materials 변환
        learner_materials = []
        for mat in dev.materials:
            learner_materials.append({
                "title": mat.title,
                "type": mat.type,
                "content": mat.description or "",
                "format": "PDF/PPT" if mat.slides or mat.pages else "기타",
            })

        development_dict = {
            # [19] 학습자용 자료 개발
            "learner_materials": learner_materials,
            # [20] 교수자용 매뉴얼 개발
            "instructor_guide": {
                "overview": dev.facilitator_manual or "",
                "session_guides": [mod.title for mod in dev.lesson_plan.modules],
                "facilitation_tips": ["학습자 참여 유도", "질문 활용"],
                "troubleshooting": ["기술적 문제 대응"],
            },
            # [21] 운영자용 매뉴얼 개발
            "operator_manual": {
                "system_setup": dev.operator_manual or "시스템 설정 가이드",
                "operation_procedures": ["등록 관리", "출석 관리"],
                "support_procedures": ["학습자 문의 대응"],
                "escalation_process": "문제 발생 시 담당자에게 보고",
            },
            # [22] 평가 도구·문항 개발
            "assessment_tools": [
                {
                    "item_id": tool,
                    "type": "평가 도구",
                    "question": tool,
                    "aligned_objective": "",
                    "scoring_criteria": "",
                }
                for tool in dev.assessment_tools
            ],
            # [23] 전문가 검토
            "expert_review": {
                "reviewers": ["내용 전문가"],
                "review_criteria": ["내용 정확성", "교수 설계 적절성"],
                "feedback_summary": dev.expert_review_plan or "",
                "revisions_made": [],
            },
        }

        # Implementation 단계 변환
        impl = self.implementation

        implementation_dict = {
            # [24] 교수자·운영자 오리엔테이션
            "instructor_orientation": {
                "orientation_objectives": ["프로그램 이해", "운영 절차 숙지"],
                "schedule": impl.orientation_plan or "사전 1주일 전",
                "materials": ["교수자 가이드", "운영 매뉴얼"],
                "competency_checklist": ["내용 이해도", "진행 능력"],
            },
            # [25] 시스템/환경 점검
            "system_check": {
                "checklist": impl.technical_requirements or ["네트워크 연결", "장비 점검"],
                "technical_validation": impl.system_check_plan or "시스템 테스트 완료",
                "contingency_plans": ["비상 대응 계획 수립"],
            },
            # [26] 프로토타입 실행
            "prototype_execution": {
                "pilot_scope": impl.pilot_execution_plan or "소규모 파일럿 테스트",
                "participants": "10명 내외",
                "execution_log": [],
                "issues_encountered": [],
            },
            # [27] 운영 모니터링 및 지원
            "monitoring": {
                "monitoring_criteria": [impl.monitoring_plan] if impl.monitoring_plan else ["학습 진도", "참여율"],
                "support_channels": ["이메일", "전화"],
                "issue_resolution_log": [],
                "real_time_adjustments": [],
            },
        }

        # Evaluation 단계 변환
        ev = self.evaluation

        # 퀴즈 아이템 변환
        quiz_tools = [
            {
                "item_id": q.id,
                "type": q.type,
                "question": q.question,
                "scoring_rubric": q.explanation or "",
            }
            for q in ev.quiz_items
        ]

        evaluation_dict = {
            # E1: 형성평가 (소항목 28-29)
            "formative": {
                # [28] 파일럿/초기 실행 중 자료 수집
                "data_collection": {
                    "methods": ["설문", "관찰", "면담"],
                    "learner_feedback": [],
                    "performance_data": {},
                    "observations": [ev.pilot_data_collection] if ev.pilot_data_collection else [],
                },
                # [29] 형성평가 결과 기반 1차 프로그램 개선
                "improvements": [
                    {
                        "issue_identified": ev.formative_improvement or "개선 필요 사항 식별",
                        "improvement_action": "개선 조치 실행",
                        "priority": "높음",
                    }
                ] if ev.formative_improvement else [],
            },
            # E2: 총괄평가 및 채택 결정 (소항목 30-32)
            "summative": {
                # [30] 총괄 평가 문항 개발
                "assessment_tools": quiz_tools,
                # [31] 총괄평가 시행 및 프로그램 효과 분석
                "effectiveness_analysis": {
                    "learning_outcomes": {},
                    "goal_achievement_rate": "",
                    "statistical_analysis": ev.summative_evaluation_plan or "",
                    "recommendations": [],
                },
                # [32] 프로그램 채택 여부 결정
                "adoption_decision": {
                    "decision": "",
                    "rationale": ev.adoption_decision_criteria or "",
                    "conditions": [],
                    "stakeholder_approval": "승인 대기",
                },
            },
            # [33] E3: 프로그램 개선 및 환류
            "improvement_plan": {
                "feedback_summary": ev.feedback_plan or "",
                "improvement_areas": [ev.program_improvement] if ev.program_improvement else [],
                "action_items": [],
                "feedback_loop": "평가 결과를 바탕으로 다음 교육 과정에 반영",
                "next_iteration_goals": [],
            },
        }

        return {
            "analysis": analysis_dict,
            "design": design_dict,
            "development": development_dict,
            "implementation": implementation_dict,
            "evaluation": evaluation_dict,
        }


class ToolCall(BaseModel):
    """도구 호출 기록"""
    step: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[int] = None
    success: bool = True
    output_data: Optional[dict[str, Any]] = None
    feedback: Optional[dict[str, Any]] = None


class Trajectory(BaseModel):
    """궤적 기록"""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)


class Metadata(BaseModel):
    """메타데이터"""
    model: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    execution_time_seconds: float = 0.0
    cost_usd: float = 0.0
    agent_version: str = "0.1.0"
    iterations: int = 0


class AgentResult(BaseModel):
    """Agent 최종 출력"""
    scenario_id: str
    agent_id: str = "eduplanner"
    timestamp: datetime = Field(default_factory=datetime.now)
    addie_output: ADDIEOutput
    trajectory: Trajectory = Field(default_factory=Trajectory)
    metadata: Metadata


class EvaluationFeedback(BaseModel):
    """평가 에이전트 피드백 (ADDIE Rubric 13항목 기반)"""
    score: float = Field(..., ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    addie_scores: dict[str, float] = Field(
        default_factory=dict,
        description="ADDIE Rubric 13항목 점수 (A1-A3, D1-D3, Dev1-Dev2, I1-I2, E1-E3)"
    )
    weighted_score: Optional[float] = Field(
        default=None,
        description="가중치 적용 점수 (0-100)"
    )
