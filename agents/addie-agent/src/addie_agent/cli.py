"""
ADDIE Agent CLI

명령줄 인터페이스를 통해 ADDIE Agent를 실행합니다.
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

from addie_agent.agent import ADDIEAgent

app = typer.Typer(
    name="addie-agent",
    help="ADDIE 모형 기반 순차적 교수설계 에이전트",
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
    """시나리오 파일을 입력받아 ADDIE 교수설계를 실행합니다.

    Example:
        addie-agent run --input scenario.json --output result.json
        addie-agent run -i scenario.json -o result.json -t trajectory.json -v
    """
    if verbose:
        console.print(Panel.fit(
            "[bold blue]ADDIE Agent[/bold blue]\n"
            "ADDIE 모형 기반 순차적 교수설계 에이전트",
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

    # 에이전트 생성 및 실행
    agent = ADDIEAgent(model=model, debug=debug)

    try:
        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("ADDIE 교수설계 실행 중...", total=None)
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

    # Trajectory 저장 (evaluator가 기대하는 형식으로 저장)
    if trajectory_file:
        trajectory_file.parent.mkdir(parents=True, exist_ok=True)
        # trajectory 데이터는 result["trajectory"]에 저장되어 있음
        result_trajectory = result.get("trajectory", {})
        result_metadata = result.get("metadata", {})
        trajectory = {
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "agent_id": "addie-agent",
            # evaluator가 기대하는 중첩 구조
            "trajectory": {
                "tool_calls": result_trajectory.get("tool_calls", []),
                "reasoning_steps": result_trajectory.get("reasoning_steps", []),
            },
            "metadata": {
                "agent_id": "addie-agent",
                "model": result_metadata.get("model", "solar-mini"),
                "execution_time_seconds": result_metadata.get("execution_time_seconds", 0),
                "tool_calls_count": result_metadata.get("tool_calls_count", 0),
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
    meta_table.add_row("모델", meta.get("model", "unknown"))

    console.print(meta_table)

    # ADDIE 산출물 요약
    addie = result.get("addie_output", {})

    summary_table = Table(title="ADDIE 산출물 요약")
    summary_table.add_column("단계", style="cyan")
    summary_table.add_column("항목", style="white")
    summary_table.add_column("수량", style="green")

    # Analysis
    analysis = addie.get("analysis", {})
    la = analysis.get("learner_analysis", {})
    ta = analysis.get("task_analysis", {})
    summary_table.add_row("Analysis", "학습자 특성", str(len(la.get("characteristics", []))))
    summary_table.add_row("", "주요 주제", str(len(ta.get("main_topics", []))))

    # Design
    design = addie.get("design", {})
    summary_table.add_row("Design", "학습 목표", str(len(design.get("learning_objectives", []))))
    strategy = design.get("instructional_strategy", {})
    summary_table.add_row("", "교수 사태", str(len(strategy.get("sequence", []))))

    # Development
    dev = addie.get("development", {})
    lp = dev.get("lesson_plan", {})
    summary_table.add_row("Development", "레슨 모듈", str(len(lp.get("modules", []))))
    summary_table.add_row("", "학습 자료", str(len(dev.get("materials", []))))

    # Implementation
    impl = addie.get("implementation", {})
    fg_len = len(impl.get("facilitator_guide", "") or "")
    lg_len = len(impl.get("learner_guide", "") or "")
    summary_table.add_row("Implementation", "진행자 가이드", f"{fg_len}자")
    summary_table.add_row("", "학습자 가이드", f"{lg_len}자")

    # Evaluation
    eval_result = addie.get("evaluation", {})
    summary_table.add_row("Evaluation", "퀴즈 문항", str(len(eval_result.get("quiz_items", []))))
    rubric = eval_result.get("rubric", {})
    summary_table.add_row("", "평가 기준", str(len(rubric.get("criteria", []))))

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
    from addie_agent import __version__
    console.print(f"ADDIE Agent v{__version__}")


if __name__ == "__main__":
    app()
