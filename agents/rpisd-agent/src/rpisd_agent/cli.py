"""
RPISD Agent CLI

래피드 프로토타이핑 교수설계 에이전트 CLI 인터페이스
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="rpisd-agent",
    help="RPISD 래피드 프로토타이핑 교수설계 에이전트",
)


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
        help="사용할 모델 (solar-mini, solar-pro-2)",
    ),
    max_iterations: int = typer.Option(
        3,
        "--max-iter",
        help="최대 루프 반복 횟수 (프로토타입/개발 각각)",
    ),
    quality_threshold: float = typer.Option(
        0.8,
        "--threshold", "-q",
        help="품질 기준 점수 (0.0-1.0)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="상세 출력 모드",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="디버그 모드",
    ),
):
    """RPISD 래피드 프로토타이핑 교수설계 에이전트를 실행합니다.

    Example:
        rpisd-agent run --input scenario.json --output result.json
        rpisd-agent run -i scenario.json -o result.json -t trajectory.json -v
    """
    from rpisd_agent.agent import RPISDAgent

    # 시나리오 로드
    if verbose:
        typer.echo(f"시나리오 로드: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    scenario_id = scenario.get("scenario_id", "unknown")
    if verbose:
        typer.echo(f"시나리오 ID: {scenario_id}")
        typer.echo(f"모델: {model}, 최대 반복: {max_iterations}, 품질 기준: {quality_threshold}")

    # 에이전트 생성 및 실행
    agent = RPISDAgent(
        model=model,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        debug=debug,
    )

    try:
        result = agent.run(scenario)
    except Exception as e:
        typer.secho(f"실행 오류: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 결과 저장 (표준 스키마: addie_output 내용만 최상위로)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    addie_output = result.get("addie_output", result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(addie_output, f, ensure_ascii=False, indent=2)

    if verbose:
        typer.echo(f"결과 저장: {output_file}")

    # Trajectory 저장 (evaluator가 기대하는 형식으로 저장)
    if trajectory_file:
        trajectory_file.parent.mkdir(parents=True, exist_ok=True)
        result_trajectory = result.get("trajectory", {})
        result_metadata = result.get("metadata", {})
        trajectory = {
            "scenario_id": scenario_id,
            "agent_id": "rpisd-agent",
            "trajectory": {
                "tool_calls": result_trajectory.get("tool_calls", []),
                "reasoning_steps": result_trajectory.get("reasoning_steps", []),
            },
            "metadata": {
                "agent_id": "rpisd-agent",
                "model": result_metadata.get("model", "solar-mini"),
                "execution_time_seconds": result_metadata.get("execution_time_seconds", 0),
                "tool_calls_count": result_metadata.get("tool_calls_count", 0),
            },
        }
        with open(trajectory_file, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        if verbose:
            typer.echo(f"Trajectory 저장: {trajectory_file}")

    # 결과 요약
    if verbose:
        metadata = result.get("metadata", {})
        typer.echo("\n" + "=" * 50)
        typer.secho("실행 완료!", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  출력 파일: {output_file}")
        typer.echo(f"  실행 시간: {metadata.get('execution_time_seconds', 0):.2f}초")
        typer.echo(f"  도구 호출: {metadata.get('tool_calls_count', 0)}회")
        typer.echo(f"  프로토타입 반복: {metadata.get('prototype_iterations', 0)}회")
        typer.echo(f"  개발 반복: {metadata.get('development_iterations', 0)}회")
        typer.echo(f"  최종 품질: {metadata.get('final_quality_score', 0):.2f}")

        if metadata.get("errors"):
            typer.secho(f"  오류: {len(metadata['errors'])}개", fg=typer.colors.YELLOW)


@app.command()
def validate(
    output_path: Path = typer.Argument(
        ...,
        help="출력 JSON 파일 경로",
        exists=True,
        readable=True,
    ),
    schema_path: Optional[Path] = typer.Option(
        None,
        "--schema", "-s",
        help="스키마 JSON 파일 경로 (기본: shared/schemas/output_schema.json)",
    ),
):
    """출력 결과를 스키마에 대해 검증합니다."""
    try:
        import jsonschema
    except ImportError:
        typer.secho("jsonschema 패키지가 필요합니다: pip install jsonschema", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 출력 파일 로드
    typer.echo(f"출력 파일 로드: {output_path}")
    with open(output_path, "r", encoding="utf-8") as f:
        output = json.load(f)

    # 스키마 로드
    if schema_path is None:
        # 기본 스키마 경로 시도
        default_schema = Path(__file__).parent.parent.parent.parent.parent / "shared" / "schemas" / "output_schema.json"
        if default_schema.exists():
            schema_path = default_schema
        else:
            typer.secho("스키마 파일을 찾을 수 없습니다. --schema 옵션을 사용하세요.", fg=typer.colors.RED)
            raise typer.Exit(1)

    typer.echo(f"스키마 로드: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # 검증
    try:
        jsonschema.validate(instance=output, schema=schema)
        typer.secho("스키마 검증 통과!", fg=typer.colors.GREEN, bold=True)
    except jsonschema.ValidationError as e:
        typer.secho(f"스키마 검증 실패: {e.message}", fg=typer.colors.RED)
        typer.echo(f"  경로: {' -> '.join(str(p) for p in e.absolute_path)}")
        raise typer.Exit(1)


@app.command()
def info():
    """에이전트 정보를 표시합니다."""
    from rpisd_agent import __version__

    typer.echo("=" * 50)
    typer.secho("RPISD Agent", fg=typer.colors.CYAN, bold=True)
    typer.echo("=" * 50)
    typer.echo(f"버전: {__version__}")
    typer.echo()
    typer.echo("래피드 프로토타이핑 기반 교수설계 에이전트")
    typer.echo()
    typer.echo("특징:")
    typer.echo("  - 이중 루프: 설계↔사용성평가, 개발↔사용성평가")
    typer.echo("  - 프로토타입 버전 관리")
    typer.echo("  - 다중 피드백 통합 (의뢰인/전문가/학습자)")
    typer.echo()
    typer.echo("도구 (14개):")
    typer.echo("  - 착수: kickoff_meeting")
    typer.echo("  - 분석: analyze_gap, analyze_performance,")
    typer.echo("          analyze_learner_characteristics, analyze_initial_task")
    typer.echo("  - 설계: design_instruction, develop_prototype, analyze_task_detailed")
    typer.echo("  - 평가: evaluate_with_client, evaluate_with_expert,")
    typer.echo("          evaluate_with_learner, aggregate_feedback")
    typer.echo("  - 개발: develop_final_program")
    typer.echo("  - 실행: implement_program")


def main():
    """CLI 엔트리포인트"""
    app()


if __name__ == "__main__":
    main()
