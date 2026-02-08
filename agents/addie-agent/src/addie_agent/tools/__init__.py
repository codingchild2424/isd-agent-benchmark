"""
ADDIE Agent Tools

ADDIE 5단계별 도구 모음 (총 14개) - Branch(2009) ADDIE 모형 기반
- Analysis: 4개 (analyze_needs, analyze_learner, analyze_context, analyze_task)
- Design: 3개 (design_objectives, design_assessment, design_strategy)
- Development: 2개 (create_lesson_plan, create_materials)
- Implementation: 2개 (create_implementation_plan, create_maintenance_plan)
- Evaluation: 3개 (create_quiz_items, create_rubric, create_program_evaluation)
"""

from addie_agent.tools.analysis import (
    analyze_needs,
    analyze_learner,
    analyze_context,
    analyze_task,
)
from addie_agent.tools.design import (
    design_objectives,
    design_assessment,
    design_strategy,
)
from addie_agent.tools.development import (
    create_lesson_plan,
    create_materials,
)
from addie_agent.tools.implementation import (
    create_implementation_plan,
    create_maintenance_plan,
)
from addie_agent.tools.evaluation import (
    create_quiz_items,
    create_rubric,
    create_program_evaluation,
)

__all__ = [
    # Analysis (요구분석, 학습자분석, 환경분석, 과제분석)
    "analyze_needs",
    "analyze_learner",
    "analyze_context",
    "analyze_task",
    # Design (목표명세화, 평가도구, 교수전략)
    "design_objectives",
    "design_assessment",
    "design_strategy",
    # Development (레슨플랜, 교수자료)
    "create_lesson_plan",
    "create_materials",
    # Implementation (실행계획, 유지관리)
    "create_implementation_plan",
    "create_maintenance_plan",
    # Evaluation (퀴즈, 루브릭, 성과평가)
    "create_quiz_items",
    "create_rubric",
    "create_program_evaluation",
]
