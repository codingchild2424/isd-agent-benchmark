"""ReAct-ISD ADDIE 도구 모음 (33개 소항목 완전 지원)"""

# Analysis 단계 도구 (ADDIE 소항목 1-10)
from react_isd.tools.analyze import (
    analyze_learners,       # 5. 학습자 분석
    analyze_context,        # 6. 환경 분석
    analyze_task,           # 7. 초기 학습목표, 8. 하위 기능 분석
    analyze_needs,          # 1-4. 요구분석 (Gap Analysis)
    analyze_entry_behavior, # 9. 출발점 행동 분석
    review_task_analysis,   # 10. 과제분석 검토·정리
)

# Design 단계 도구 (ADDIE 소항목 11-18)
from react_isd.tools.design import (
    design_objectives,      # 11. 학습목표 정교화
    design_assessment,      # 12. 평가 계획 수립
    design_strategy,        # 14. 교수적 전략
    design_content,         # 13. 교수 내용 선정
    design_non_instructional,  # 15. 비교수적 전략
    design_media,           # 16. 매체 선정
    design_storyboard,      # 18. 스토리보드/화면 흐름
)

# Development 단계 도구 (ADDIE 소항목 17, 19-23)
from react_isd.tools.develop import (
    create_lesson_plan,     # 17. 학습활동 시간 구조화
    create_materials,       # 19. 학습자용 자료 개발
    create_facilitator_manual,  # 20. 교수자용 매뉴얼
    create_operator_manual,     # 21. 운영자용 매뉴얼
    create_expert_review,       # 23. 전문가 검토
)

# Implementation 단계 도구 (ADDIE 소항목 24-27)
from react_isd.tools.implement import (
    create_implementation_plan,  # 26. 프로토타입 실행 (기본)
    create_orientation_plan,     # 24. 교수자 오리엔테이션
    create_system_checklist,     # 25. 시스템/환경 점검
    create_pilot_plan,           # 26. 프로토타입 실행 (상세)
    create_monitoring_plan,      # 27. 운영 모니터링
)

# Evaluation 단계 도구 (ADDIE 소항목 22, 28-33)
from react_isd.tools.evaluate import (
    create_quiz_items,          # 22. 평가 문항 개발
    create_rubric,              # 30. 총괄 평가 문항
    create_data_collection_plan,   # 28. 자료 수집
    create_formative_improvement,  # 29. 형성평가 기반 개선
    create_program_evaluation,     # 30-33. 총괄평가, 채택결정, 효과분석, 유지관리
)

__all__ = [
    # Analysis tools (6개)
    "analyze_learners",
    "analyze_context",
    "analyze_task",
    "analyze_needs",
    "analyze_entry_behavior",
    "review_task_analysis",
    # Design tools (7개)
    "design_objectives",
    "design_assessment",
    "design_strategy",
    "design_content",
    "design_non_instructional",
    "design_media",
    "design_storyboard",
    # Development tools (5개)
    "create_lesson_plan",
    "create_materials",
    "create_facilitator_manual",
    "create_operator_manual",
    "create_expert_review",
    # Implementation tools (5개)
    "create_implementation_plan",
    "create_orientation_plan",
    "create_system_checklist",
    "create_pilot_plan",
    "create_monitoring_plan",
    # Evaluation tools (5개)
    "create_quiz_items",
    "create_rubric",
    "create_data_collection_plan",
    "create_formative_improvement",
    "create_program_evaluation",
]
