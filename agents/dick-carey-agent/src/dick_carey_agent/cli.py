"""
Dick & Carey Agent CLI

명령줄 인터페이스를 통해 Dick & Carey Agent를 실행합니다.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from dick_carey_agent.agent import DickCareyAgent

app = typer.Typer(
    name="dick-carey-agent",
    help="Dick & Carey 모형 기반 체제적 교수설계 에이전트",
)
console = Console()


@app.command()
def run(
    input_file: Path = typer.Option(
        ...,
        "--input", "-i",
        help="입력 시나리오 JSON 파일 경로",
        exists=True,
        readable=True,
    ),
    output_file: Path = typer.Option(
        ...,
        "--output", "-o",
        help="출력 결과 JSON 파일 경로",
    ),
    trajectory_file: Optional[Path] = typer.Option(
        None,
        "--trajectory", "-t",
        help="궤적(trajectory) JSON 파일 경로 (선택)",
    ),
    model: str = typer.Option(
        "solar-mini",
        "--model", "-m",
        help="사용할 LLM 모델",
    ),
    max_iterations: int = typer.Option(
        3,
        "--max-iterations",
        help="최대 형성평가-수정 반복 횟수",
    ),
    quality_threshold: float = typer.Option(
        7.0,
        "--quality-threshold",
        help="품질 기준 점수 (0-10)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="상세 출력 모드",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="디버그 모드 활성화",
    ),
):
    """시나리오 파일을 입력받아 Dick & Carey 교수설계를 실행합니다.

    Example:
        dick-carey-agent run --input scenario.json --output result.json
        dick-carey-agent run -i scenario.json -o result.json -t trajectory.json -v
        dick-carey-agent run -i scenario.json -o result.json --max-iterations 5 --quality-threshold 8.0
    """
    if verbose:
        console.print(Panel.fit(
            "[bold blue]Dick & Carey Agent[/bold blue]\n"
            "체제적 교수설계(Systems Approach) 10단계 에이전트\n"
            f"피드백 루프: 최대 {max_iterations}회, 품질 기준 {quality_threshold}점",
            border_style="blue",
        ))

    # 시나리오 로드
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            scenario = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]오류: JSON 파싱 실패 - {e}[/red]")
        raise typer.Exit(1)

    if verbose:
        console.print(f"\n[cyan]시나리오:[/cyan] {scenario.get('title', scenario.get('scenario_id', 'unknown'))}")
        console.print(f"[cyan]모델:[/cyan] {model}")
        console.print(f"[cyan]품질 기준:[/cyan] {quality_threshold}점")
        console.print(f"[cyan]최대 반복:[/cyan] {max_iterations}회")

    # 에이전트 생성 및 실행
    agent = DickCareyAgent(
        model=model,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        debug=debug,
    )

    try:
        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Dick & Carey 교수설계 실행 중...", total=None)
                result = agent.run(scenario)
                progress.update(task, description="완료!")
        else:
            result = agent.run(scenario)
    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")
        raise typer.Exit(1)

    # 결과 요약 출력
    if verbose:
        _print_summary(result)

    # 결과 저장 (표준 스키마: addie_output 내용만 최상위로)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    addie_output = result.get("addie_output", result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(addie_output, f, ensure_ascii=False, indent=2, default=str)

    if verbose:
        console.print(f"\n[green]결과 저장:[/green] {output_file}")

    # Trajectory 저장
    if trajectory_file:
        trajectory_file.parent.mkdir(parents=True, exist_ok=True)
        result_trajectory = result.get("trajectory", {})
        result_metadata = result.get("metadata", {})
        trajectory = {
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "agent_id": "dick-carey-agent",
            "trajectory": {
                "tool_calls": result_trajectory.get("tool_calls", []),
                "reasoning_steps": result_trajectory.get("reasoning_steps", []),
            },
            "metadata": {
                "agent_id": "dick-carey-agent",
                "model": result_metadata.get("model", "solar-mini"),
                "execution_time_seconds": result_metadata.get("execution_time_seconds", 0),
                "tool_calls_count": result_metadata.get("tool_calls_count", 0),
                "iteration_count": result_metadata.get("iteration_count", 0),
            },
        }
        with open(trajectory_file, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2, default=str)

        if verbose:
            console.print(f"[green]Trajectory 저장:[/green] {trajectory_file}")


def _print_summary(result: dict):
    """결과 요약 출력"""
    console.print("\n")

    # 메타데이터 테이블
    meta = result.get("metadata", {})
    meta_table = Table(title="실행 정보", show_header=False)
    meta_table.add_column("항목", style="cyan")
    meta_table.add_column("값", style="white")

    meta_table.add_row("실행 시간", f"{meta.get('execution_time_seconds', 0):.2f}초")
    meta_table.add_row("도구 호출", f"{meta.get('tool_calls_count', 0)}회")
    meta_table.add_row("형성평가 반복", f"{meta.get('iteration_count', 0)}회")
    meta_table.add_row("최종 품질 점수", f"{meta.get('final_quality_score', 0):.1f}점")
    meta_table.add_row("최종 결정", meta.get("final_decision", ""))
    meta_table.add_row("모델", meta.get("model", "unknown"))

    console.print(meta_table)

    # Dick & Carey 산출물 요약
    dc = result.get("dick_carey_output", {})

    summary_table = Table(title="Dick & Carey 10단계 산출물 요약")
    summary_table.add_column("단계", style="cyan")
    summary_table.add_column("항목", style="white")
    summary_table.add_column("수량/값", style="green")

    # 1단계: 교수목적
    goal = dc.get("goal", {})
    summary_table.add_row("1. 교수목적", "목적 진술", goal.get("goal_statement", "")[:40] + "..." if goal.get("goal_statement") else "-")

    # 2단계: 교수분석
    analysis = dc.get("instructional_analysis", {})
    summary_table.add_row("2. 교수분석", "하위 기능", str(len(analysis.get("sub_skills", []))))

    # 3단계: 학습자/환경
    lc = dc.get("learner_context", {})
    learner = lc.get("learner", {})
    summary_table.add_row("3. 학습자/환경", "출발점 행동", str(len(learner.get("entry_behaviors", []))))

    # 4단계: 수행목표
    objectives = dc.get("performance_objectives", {})
    summary_table.add_row("4. 수행목표", "가능 목표", str(len(objectives.get("enabling_objectives", []))))

    # 5단계: 평가도구
    assessment = dc.get("assessment_instruments", {})
    summary_table.add_row("5. 평가도구", "사후평가 문항", str(len(assessment.get("post_test", []))))

    # 6단계: 교수전략
    strategy = dc.get("instructional_strategy", {})
    summary_table.add_row("6. 교수전략", "전달 방법", strategy.get("delivery_method", "-"))

    # 7단계: 교수자료
    materials = dc.get("instructional_materials", {})
    summary_table.add_row("7. 교수자료", "학습자 자료", str(len(materials.get("learner_materials", []))))
    summary_table.add_row("", "슬라이드", str(len(materials.get("slide_contents", []))))

    # 8단계: 형성평가
    formative = dc.get("formative_evaluation", {})
    summary_table.add_row("8. 형성평가", "품질 점수", f"{formative.get('quality_score', 0):.1f}점")

    # 9단계: 수정
    revision_log = dc.get("revision_log", [])
    summary_table.add_row("9. 수정", "수정 횟수", str(len(revision_log)))

    # 10단계: 총괄평가
    summative = dc.get("summative_evaluation", {})
    summary_table.add_row("10. 총괄평가", "효과성 점수", f"{summative.get('effectiveness_score', 0):.1f}점")
    summary_table.add_row("", "결정", summative.get("decision", "-"))

    console.print(summary_table)

    # 오류 표시
    errors = meta.get("errors", [])
    if errors:
        console.print("\n[yellow]경고:[/yellow]")
        for error in errors:
            console.print(f"  - {error}")


@app.command()
def validate(
    scenario_file: Path = typer.Argument(
        ...,
        help="시나리오 JSON 파일 경로",
        exists=True,
        readable=True,
    ),
):
    """시나리오 파일의 유효성을 검증합니다."""
    try:
        with open(scenario_file, "r", encoding="utf-8") as f:
            scenario = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]오류: JSON 파싱 실패 - {e}[/red]")
        raise typer.Exit(1)

    # 필수 필드 검증
    required_fields = ["scenario_id", "context", "learning_goals"]
    missing = [f for f in required_fields if f not in scenario]

    if missing:
        console.print(f"[red]누락된 필수 필드: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    context = scenario.get("context", {})
    context_fields = ["target_audience", "duration", "learning_environment"]
    missing_context = [f for f in context_fields if f not in context]

    if missing_context:
        console.print(f"[yellow]경고 - context 누락 필드: {', '.join(missing_context)}[/yellow]")

    console.print(f"[green]✓ 유효한 시나리오 파일입니다.[/green]")
    console.print(f"  시나리오 ID: {scenario.get('scenario_id')}")
    console.print(f"  제목: {scenario.get('title', 'N/A')}")
    console.print(f"  학습 목표: {len(scenario.get('learning_goals', []))}개")


@app.command()
def version():
    """버전 정보를 출력합니다."""
    from dick_carey_agent import __version__
    console.print(f"Dick & Carey Agent v{__version__}")
    console.print("체제적 교수설계(Systems Approach) 10단계 에이전트")
    console.print("Dick, W., Carey, L., & Carey, J. O. (2009) 기반")


if __name__ == "__main__":
    app()
