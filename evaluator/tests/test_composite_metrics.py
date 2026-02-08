"""Composite 평가기 단위 테스트 (ADDIE 모델 기반)"""
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "evaluator" / "src"))

from isd_evaluator.models import (
    ADDIEPhase, ADDIEScore, CompositeScore, PhaseScore, RubricItem, TrajectoryScore
)


def _create_sample_phase_score(phase: ADDIEPhase, base_score: float = 8.0) -> PhaseScore:
    """테스트용 PhaseScore 생성"""
    items = [
        RubricItem(
            item_id=f"{phase.value[:1].upper()}1",
            phase=phase,
            name=f"{phase.value} 테스트 항목",
            score=base_score,
            reasoning="테스트용 평가"
        )
    ]
    return PhaseScore(
        phase=phase,
        items=items,
        raw_score=base_score,
        weighted_score=base_score * 2,
        max_score=10.0
    )


def _create_sample_addie_score(base_score: float = 8.0) -> ADDIEScore:
    """테스트용 ADDIEScore 생성"""
    return ADDIEScore(
        analysis=_create_sample_phase_score(ADDIEPhase.ANALYSIS, base_score),
        design=_create_sample_phase_score(ADDIEPhase.DESIGN, base_score),
        development=_create_sample_phase_score(ADDIEPhase.DEVELOPMENT, base_score),
        implementation=_create_sample_phase_score(ADDIEPhase.IMPLEMENTATION, base_score),
        evaluation=_create_sample_phase_score(ADDIEPhase.EVALUATION, base_score),
        total_raw=base_score * 5,
        total_weighted=base_score * 10,
        normalized_score=base_score * 10,
        strengths=["테스트 강점"],
        improvements=["테스트 개선점"],
        overall_assessment="테스트 종합평가"
    )


def test_addie_score_creation():
    """ADDIEScore 생성 테스트"""
    print("Testing ADDIEScore creation...", end=" ")

    score = _create_sample_addie_score(8.0)
    assert score.normalized_score == 80.0
    assert len(score.phases) == 5

    print("PASS")


def test_composite_score_total_with_trajectory():
    """복합 점수 계산 테스트 (궤적 포함)"""
    print("Testing composite score with trajectory...", end=" ")

    addie = _create_sample_addie_score(8.0)  # normalized_score = 80

    trajectory = TrajectoryScore(
        tool_correctness=25.0,
        argument_accuracy=24.0,
        redundancy_avoidance=24.5,
        result_utilization=23.0  # total = 96.5
    )

    score = CompositeScore(
        addie=addie,
        trajectory=trajectory,
        addie_weight=0.7,
        trajectory_weight=0.3
    )

    # Composite total = 80 * 0.7 + 96.5 * 0.3 = 56 + 28.95 = 84.95
    expected_total = 80 * 0.7 + 96.5 * 0.3

    actual = score.total
    assert abs(actual - expected_total) < 0.01, f"Expected {expected_total}, got {actual}"

    print("PASS")


def test_composite_score_total_without_trajectory():
    """복합 점수 계산 테스트 (궤적 미포함)"""
    print("Testing composite score without trajectory...", end=" ")

    addie = _create_sample_addie_score(9.0)  # normalized_score = 90

    score = CompositeScore(
        addie=addie,
        trajectory=None,
        addie_weight=0.7,
        trajectory_weight=0.3
    )

    # 궤적이 없으면 ADDIE 점수만 반환
    expected_total = 90.0

    actual = score.total
    assert actual == expected_total, f"Expected {expected_total}, got {actual}"

    print("PASS")


def test_trajectory_score_total():
    """궤적 점수 합계 테스트"""
    print("Testing trajectory score total...", end=" ")

    trajectory = TrajectoryScore(
        tool_correctness=20.0,
        argument_accuracy=21.0,
        redundancy_avoidance=22.0,
        result_utilization=23.0
    )

    expected = 20 + 21 + 22 + 23  # = 86
    assert trajectory.total == expected, f"Expected {expected}, got {trajectory.total}"

    print("PASS")


def test_composite_score_properties():
    """복합 점수 속성 테스트"""
    print("Testing CompositeScore properties...", end=" ")

    addie = _create_sample_addie_score(8.5)

    trajectory = TrajectoryScore(
        tool_correctness=25.0,
        argument_accuracy=24.0,
        redundancy_avoidance=24.5,
        result_utilization=23.0
    )

    score = CompositeScore(addie=addie, trajectory=trajectory)

    # addie_score property
    assert score.addie_score == 85.0, f"Expected 85.0, got {score.addie_score}"

    # trajectory_score property
    assert score.trajectory_score == 96.5, f"Expected 96.5, got {score.trajectory_score}"

    print("PASS")


def test_weight_impact_analysis():
    """가중치 영향 분석 테스트"""
    print("Testing weight impact analysis...", end=" ")

    # 산출물 점수가 높고 과정 점수가 낮은 경우
    addie = _create_sample_addie_score(9.5)  # normalized_score = 95

    trajectory = TrajectoryScore(
        tool_correctness=15.0,
        argument_accuracy=15.0,
        redundancy_avoidance=15.0,
        result_utilization=15.0  # total = 60
    )

    # 70:30 가중치
    score_70_30 = CompositeScore(
        addie=addie,
        trajectory=trajectory,
        addie_weight=0.7,
        trajectory_weight=0.3
    )

    # 50:50 가중치
    score_50_50 = CompositeScore(
        addie=addie,
        trajectory=trajectory,
        addie_weight=0.5,
        trajectory_weight=0.5
    )

    # 산출물이 높을 때 70:30이 유리해야 함
    assert score_70_30.total > score_50_50.total, "70:30 should favor high output scores"

    print("PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Composite 평가기 단위 테스트 (ADDIE 모델 기반)")
    print("=" * 60)

    tests = [
        test_addie_score_creation,
        test_composite_score_total_with_trajectory,
        test_composite_score_total_without_trajectory,
        test_trajectory_score_total,
        test_composite_score_properties,
        test_weight_impact_analysis,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL - {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"결과: {passed}/{len(tests)} 테스트 통과")

    if failed > 0:
        sys.exit(1)
