"""
Dick & Carey Agent Tools

Dick & Carey 모형의 10단계에 대응하는 11개 도구입니다.

단계별 도구 구성:
- 1단계: set_instructional_goal
- 2단계: analyze_instruction
- 3단계: analyze_entry_behaviors, analyze_context
- 4단계: write_performance_objectives
- 5단계: develop_assessment_instruments
- 6단계: develop_instructional_strategy
- 7단계: develop_instructional_materials
- 8단계: conduct_formative_evaluation
- 9단계: revise_instruction
- 10단계: conduct_summative_evaluation
"""

from dick_carey_agent.tools.goal_analysis import (
    set_instructional_goal,
    analyze_instruction,
    analyze_entry_behaviors,
    analyze_context,
)

from dick_carey_agent.tools.objective_assessment import (
    write_performance_objectives,
    develop_assessment_instruments,
)

from dick_carey_agent.tools.strategy_materials import (
    develop_instructional_strategy,
    develop_instructional_materials,
)

from dick_carey_agent.tools.evaluation import (
    conduct_formative_evaluation,
    revise_instruction,
    conduct_summative_evaluation,
)

__all__ = [
    # 1-3단계
    "set_instructional_goal",
    "analyze_instruction",
    "analyze_entry_behaviors",
    "analyze_context",
    # 4-5단계
    "write_performance_objectives",
    "develop_assessment_instruments",
    # 6-7단계
    "develop_instructional_strategy",
    "develop_instructional_materials",
    # 8-10단계
    "conduct_formative_evaluation",
    "revise_instruction",
    "conduct_summative_evaluation",
]
