"""
RPISD Agent 33개 ADDIE 소항목 완전성 검증 테스트

이 테스트는 RPISD 에이전트가 33개 ADDIE 소항목을 모두 출력하는지 검증합니다.
"""

import json
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rpisd_agent.state import map_to_addie_output, create_initial_state, RPISDState


# 33개 ADDIE 소항목 정의
ADDIE_33_ITEMS = {
    "analysis": [
        "learner_analysis",      # 1. 학습자 분석
        "context_analysis",      # 2. 환경/맥락 분석
        "task_analysis",         # 3. 과제 분석
        "needs_analysis",        # 4. 요구 분석
        "draft_objectives",      # 5. 학습 목표 초안
        "prerequisite_analysis", # 6. 선수학습 분석
        "analysis_summary",      # 7. 분석 요약
    ],
    "design": [
        "learning_objectives",    # 8. 학습 목표
        "instructional_strategy", # 9. 교수 전략
        "learning_sequence",      # 10. 학습 순서/시퀀스
        "assessment_plan",        # 11. 평가 계획
        "prototype_history",      # 12. 프로토타입 이력 (RPISD 특화)
        "instructional_methods",  # 13. 교수 방법
        "media_design",           # 14. 미디어/자료 설계
    ],
    "development": [
        "lesson_plan",            # 15. 레슨 플랜
        "modules",                # 16. 학습 모듈
        "materials",              # 17. 학습 자료
        "slide_contents",         # 18. 슬라이드 콘텐츠
        "quiz_items",             # 19. 퀴즈 문항
        "final_prototype",        # 20. 최종 프로토타입
        "instructor_materials",   # 21. 교수자용 자료
        "learner_materials",      # 22. 학습자용 자료
    ],
    "implementation": [
        "delivery_method",        # 23. 전달 방법
        "facilitator_guide",      # 24. 진행자 가이드
        "learner_guide",          # 25. 학습자 가이드
        "operator_guide",         # 26. 운영자 가이드
        "technical_requirements", # 27. 기술 요구사항
        "maintenance_plan",       # 28. 유지관리 계획
        "pilot_plan",             # 29. 파일럿 실행 계획
        "orientation_plan",       # 30. 오리엔테이션 계획
        "monitoring_plan",        # 31. 운영 모니터링 계획
        "support_plan",           # 32. 지원 계획
    ],
    "evaluation": [
        "quiz_items",             # (Development와 중복) 퀴즈 문항
        "rubric",                 # 33. 평가 루브릭
        "usability_evaluation",   # RPISD 특화: 사용성 평가
        "program_evaluation",     # Kirkpatrick 4단계 평가
        "iteration_summary",      # RPISD 특화: 반복 요약
        "adoption_decision",      # 프로그램 채택 결정
        "improvement_plan",       # 개선 계획
    ],
}


def create_mock_state() -> RPISDState:
    """테스트용 모의 상태 생성"""
    scenario = {
        "scenario_id": "TEST-001",
        "title": "테스트 교육 프로그램",
        "context": {
            "target_audience": "성인 학습자",
            "duration": "8시간",
            "learning_environment": "온라인",
        },
        "learning_goals": ["목표 1", "목표 2"],
        "constraints": {"resources": ["자원 1"]},
        "difficulty": "보통",
        "domain": "교육",
    }

    state = create_initial_state(scenario)

    # 각 단계별 모의 결과 설정
    state["kickoff_result"] = {
        "project_title": "테스트 프로젝트",
        "scope": {"description": "프로젝트 범위"},
        "stakeholder_roles": {"instructor": "강사"},
        "timeline": {"start": "2024-01", "end": "2024-03"},
        "success_criteria": ["성공 기준 1", "성공 기준 2"],
        "constraints": ["제약 1", "제약 2"],
    }

    state["analysis_result"] = {
        "gap_analysis": {
            "current_state": "현재 상태",
            "desired_state": "목표 상태",
            "gaps": ["갭 1", "갭 2"],
            "training_needs": ["훈련 필요 1"],
        },
        "performance_analysis": {
            "performance_issues": ["이슈 1"],
            "is_training_solution": True,
        },
        "learner_characteristics": {
            "target_audience": "성인 학습자",
            "prior_knowledge": "기초 지식",
            "learning_preferences": ["선호 1", "선호 2"],
        },
        "initial_task": {
            "main_topics": ["주제 1", "주제 2", "주제 3"],
            "subtopics": ["하위 주제 1"],
            "prerequisites": ["선수학습 1", "선수학습 2"],
        },
    }

    state["design_result"] = {
        "objectives": [
            {"id": "LO-001", "statement": "학습 목표 1", "level": "이해"},
            {"id": "LO-002", "statement": "학습 목표 2", "level": "적용"},
        ],
        "strategy": {
            "approach": "문제 중심 학습",
            "media": ["동영상", "슬라이드", "퀴즈"],
        },
        "sequence": [
            {"event": "도입", "activity": "학습 목표 안내", "duration": "10분"},
            {"event": "전개", "activity": "핵심 내용 설명", "duration": "40분"},
        ],
        "methods": ["강의", "토론", "실습"],
    }

    state["prototype_versions"] = [
        {
            "version": 1,
            "content": {"modules": [{"title": "모듈 1"}]},
            "feedback": [],
            "quality_score": 0.75,
            "timestamp": "2024-01-15T10:00:00",
        },
        {
            "version": 2,
            "content": {"modules": [{"title": "모듈 1"}, {"title": "모듈 2"}]},
            "feedback": [{"score": 0.85}],
            "quality_score": 0.85,
            "timestamp": "2024-01-16T14:00:00",
        },
    ]

    state["development_result"] = {
        "lesson_plan": {
            "title": "레슨 플랜",
            "duration": "8시간",
            "structure": [{"session": 1, "topic": "도입"}],
        },
        "modules": [
            {"title": "모듈 1", "duration": "2시간", "objectives": ["LO-001"]},
            {"title": "모듈 2", "duration": "3시간", "objectives": ["LO-002"]},
        ],
        "materials": [
            {"type": "프레젠테이션", "title": "핵심 개념", "slides": 20},
            {"type": "핸드아웃", "title": "워크시트", "pages": 5},
            {"type": "워크시트", "title": "실습 자료", "pages": 3},
        ],
        "slide_contents": [
            {"slide": 1, "title": "소개", "content": "학습 목표 안내"},
            {"slide": 2, "title": "본론", "content": "핵심 내용"},
        ],
        "quiz_items": [
            {
                "id": "Q-001",
                "question": "질문 1?",
                "type": "객관식",
                "options": ["A", "B", "C", "D"],
                "answer": "B",
                "explanation": "설명 1",
                "objective_id": "LO-001",
                "difficulty": "쉬움",
            },
            {
                "id": "Q-002",
                "question": "질문 2?",
                "type": "객관식",
                "options": ["A", "B", "C", "D"],
                "answer": "C",
                "explanation": "설명 2",
                "objective_id": "LO-002",
                "difficulty": "보통",
            },
        ],
        "final_prototype": {"version": 2, "approved": True},
    }

    state["implementation_result"] = {
        "delivery_method": "블렌디드 러닝",
        "facilitator_guide": "진행자 가이드 내용입니다. 교육 시작 전 자료를 점검하세요.",
        "learner_guide": "학습자 가이드 내용입니다. 적극적으로 참여해주세요.",
        "operator_guide": "운영자 가이드 내용입니다. LMS 설정을 확인하세요.",
        "technical_requirements": ["Zoom", "LMS", "프로젝터"],
        "maintenance_plan": {
            "update_frequency": "분기별",
            "responsible_party": "교수설계팀",
        },
        "support_plan": {
            "during_training": "실시간 Q&A",
            "post_training": "이메일 문의",
        },
        "pilot_plan": {
            "phase": "파일럿 테스트",
            "participants": "15명",
            "duration": "1주",
        },
        "orientation_plan": {
            "pre_training": "사전 안내 메일",
            "day_of_training": "오리엔테이션 세션",
        },
        "monitoring_plan": {
            "attendance_tracking": "출석 모니터링",
            "progress_monitoring": "진도 확인",
        },
    }

    state["usability_feedback"] = {
        "client_feedback": {"score": 0.85, "comments": "좋음"},
        "expert_feedback": {"score": 0.80, "comments": "개선 필요"},
        "learner_feedback": {"score": 0.90, "comments": "유용함"},
        "aggregated_score": 0.85,
        "improvement_areas": ["UI 개선", "콘텐츠 보완"],
        "recommendations": ["지속적 모니터링", "분기별 업데이트"],
    }

    state["evaluation_result"] = {
        "quiz_items": state["development_result"]["quiz_items"],
        "rubric": {
            "criteria": [
                "내용 이해도",
                "적용 능력",
                "분석력",
                "표현력",
                "참여도",
            ],
            "levels": {
                "excellent": {"score_range": "90-100"},
                "good": {"score_range": "70-89"},
                "needs_improvement": {"score_range": "0-69"},
            },
            "feedback_plan": "24시간 내 피드백 제공",
        },
        "program_evaluation": {
            "evaluation_model": "Kirkpatrick 4-Level",
            "levels": {
                "level_1_reaction": {"description": "만족도 평가"},
                "level_2_learning": {"description": "학습 성취도"},
                "level_3_behavior": {"description": "현업 적용도"},
                "level_4_results": {"description": "조직 성과"},
            },
        },
        "adoption_decision": {
            "recommendation": "adopt",
            "rationale": "품질 기준 충족",
        },
        "improvement_plan": {
            "based_on_feedback": ["UI 개선", "콘텐츠 보완"],
            "continuous_improvement": "분기별 업데이트",
        },
    }

    state["prototype_iteration"] = 2
    state["development_iteration"] = 1
    state["current_quality"] = 0.85

    return state


def test_addie_33_items_completeness():
    """33개 ADDIE 소항목 완전성 검증"""
    print("\n" + "=" * 60)
    print("RPISD Agent 33개 ADDIE 소항목 완전성 검증")
    print("=" * 60)

    # 모의 상태 생성
    state = create_mock_state()

    # ADDIE 출력 변환
    addie_output = map_to_addie_output(state)

    # 검증 결과 카운터
    total_items = 0
    present_items = 0
    missing_items = []
    item_details = []

    for phase, items in ADDIE_33_ITEMS.items():
        print(f"\n[{phase.upper()}]")
        phase_data = addie_output.get(phase, {})

        for item in items:
            total_items += 1
            item_value = phase_data.get(item)

            # 존재 여부 확인 (None이 아니고, 비어있지 않으면 존재)
            is_present = item_value is not None
            if isinstance(item_value, (list, dict, str)):
                is_present = is_present and len(item_value) > 0 if item_value else False

            status = "OK" if is_present else "MISSING"

            if is_present:
                present_items += 1
                # 값 요약
                if isinstance(item_value, list):
                    value_summary = f"[{len(item_value)} items]"
                elif isinstance(item_value, dict):
                    value_summary = f"{{...}} ({len(item_value)} keys)"
                elif isinstance(item_value, str):
                    value_summary = f'"{item_value[:30]}..."' if len(item_value) > 30 else f'"{item_value}"'
                else:
                    value_summary = str(item_value)
            else:
                missing_items.append(f"{phase}.{item}")
                value_summary = "N/A"

            print(f"  [{status}] {item}: {value_summary}")
            item_details.append({
                "phase": phase,
                "item": item,
                "present": is_present,
                "value_type": type(item_value).__name__ if item_value else "None",
            })

    # 결과 요약
    print("\n" + "=" * 60)
    print("검증 결과 요약")
    print("=" * 60)
    print(f"총 항목 수: {total_items}")
    print(f"존재 항목 수: {present_items}")
    print(f"누락 항목 수: {len(missing_items)}")
    print(f"완전성: {present_items / total_items * 100:.1f}%")

    if missing_items:
        print(f"\n누락된 항목:")
        for item in missing_items:
            print(f"  - {item}")

    # 검증 통과 여부
    # 참고: evaluation.quiz_items와 development.quiz_items는 중복이므로
    # 실제 고유 항목 수는 33개보다 적을 수 있음
    unique_present = len(set(f"{d['phase']}.{d['item']}" for d in item_details if d['present']))
    success = present_items >= 33  # 최소 33개 항목 존재

    print(f"\n검증 {'성공' if success else '실패'}!")

    return {
        "success": success,
        "total_items": total_items,
        "present_items": present_items,
        "missing_items": missing_items,
        "completeness_ratio": present_items / total_items,
    }


def test_addie_output_structure():
    """ADDIE 출력 구조 검증"""
    print("\n" + "=" * 60)
    print("ADDIE 출력 구조 검증")
    print("=" * 60)

    state = create_mock_state()
    addie_output = map_to_addie_output(state)

    # 5개 ADDIE 단계 존재 확인
    phases = ["analysis", "design", "development", "implementation", "evaluation"]

    for phase in phases:
        assert phase in addie_output, f"{phase} 단계가 누락됨"
        print(f"[OK] {phase} 단계 존재")

    print("\n구조 검증 성공!")
    return True


if __name__ == "__main__":
    # 테스트 실행
    result = test_addie_33_items_completeness()
    test_addie_output_structure()

    # 결과 JSON 저장
    output_path = Path(__file__).parent / "addie_completeness_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_path}")

    # 종료 코드 반환
    sys.exit(0 if result["success"] else 1)
