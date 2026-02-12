#!/usr/bin/env python3
"""
Train/Test 데이터셋 분리 스크립트

전체 벤치마크 데이터셋(idld_aligned + context_variant)을 Train(95%)과 Test(5%)로 분리합니다.
층화 추출(Stratified Sampling)을 통해 9개 축의 분포를 유지합니다.

사용법:
    python split_train_test.py [--dry-run] [--seed 42]

층화 추출(Stratified Sampling)을 통해 분포를 유지합니다.
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# 설정
SCENARIOS_DIR = Path(__file__).parent
SOURCE_DIRS = [
    SCENARIOS_DIR / "idld_aligned",
    SCENARIOS_DIR / "context_variant",
]
TRAIN_DIR = SCENARIOS_DIR / "train"
TEST_DIR = SCENARIOS_DIR / "test"
METADATA_FILE = SCENARIOS_DIR / "split_metadata.json"
REPORT_DIR = SCENARIOS_DIR / "quality_reports"

# 층화 추출에 사용할 9개 축
STRATIFY_AXES = [
    "context.duration",
    "context.learning_environment",
    "context.class_size",
    "context.institution_type",
    "context.learner_age",
    "context.learner_education",
    "context.domain_expertise",
    "domain",
    "difficulty",
]

# 분리 설정
TEST_RATIO = 0.05  # 5%
RANDOM_SEED = 42
MAX_DISTRIBUTION_DIFF = 5.0  # 분포 차이 허용치 (%)


def get_nested_value(data: dict, key: str) -> Any:
    """중첩 키로 값 추출 (예: 'context.learner_role')"""
    keys = key.split(".")
    value = data
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return None
    return value


def load_all_scenarios() -> list[tuple[Path, dict]]:
    """모든 시나리오 로드 (idld_aligned + context_variant)"""
    scenarios = []
    for source_dir in SOURCE_DIRS:
        # 재귀적으로 모든 json 파일 탐색 (context_variant는 하위 폴더 있음)
        for filepath in sorted(source_dir.rglob("*.json")):
            with open(filepath, "r", encoding="utf-8") as f:
                scenario = json.load(f)
                scenarios.append((filepath, scenario))
    return scenarios


def create_stratify_key(scenario: dict) -> str:
    """
    층화 키 생성

    9개 축을 모두 사용하면 조합이 너무 많아져서 희소해짐.
    주요 3개 축(domain, difficulty, learning_environment)으로 기본 층화하고,
    나머지 축은 사후 검증으로 분포 확인.
    """
    domain = scenario.get("domain", "unknown")
    difficulty = scenario.get("difficulty", "unknown")[:4]  # 앞 4글자로 축약
    env = get_nested_value(scenario, "context.learning_environment") or "unknown"
    env = env[:4]  # 앞 4글자로 축약

    return f"{domain}|{difficulty}|{env}"


def stratified_split(
    scenarios: list[tuple[Path, dict]],
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
    min_test_per_stratum: int = 1,
) -> tuple[list[tuple[Path, dict]], list[tuple[Path, dict]]]:
    """
    층화 추출로 Train/Test 분리

    Args:
        scenarios: (파일경로, 시나리오) 튜플 리스트
        test_ratio: 테스트셋 비율
        seed: 랜덤 시드
        min_test_per_stratum: 각 층(stratum)에서 최소 테스트 개수

    Returns:
        (train_list, test_list) 튜플
    """
    random.seed(seed)

    # 층(stratum)별로 그룹화
    strata: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for item in scenarios:
        key = create_stratify_key(item[1])
        strata[key].append(item)

    train_set = []
    test_set = []

    # 층별로 분리
    for stratum_key, items in strata.items():
        n_items = len(items)
        n_test = max(min_test_per_stratum, int(n_items * test_ratio))

        # 희소 조합 처리: 원본이 너무 적으면 전체를 train에
        # 단, 최소 2개 이상이면 1개는 test에 포함 (이슈 요구사항)
        if n_items <= 2:
            if n_items == 2:
                # 1개는 test, 1개는 train
                shuffled = items.copy()
                random.shuffle(shuffled)
                test_set.append(shuffled[0])
                train_set.append(shuffled[1])
            else:
                # 1개뿐이면 train에만 (희소 케이스 보존)
                train_set.extend(items)
            continue

        # 일반적인 경우: 비율에 따라 분리
        n_test = min(n_test, n_items - 1)  # 최소 1개는 train에 남김
        shuffled = items.copy()
        random.shuffle(shuffled)

        for i, item in enumerate(shuffled):
            if i < n_test:
                test_set.append(item)
            else:
                train_set.append(item)

    return train_set, test_set


def calculate_distribution(scenarios: list[tuple[Path, dict]], axis: str) -> dict[str, float]:
    """축별 분포 계산 (비율로 반환)"""
    values = [get_nested_value(s[1], axis) for s in scenarios]
    counter = Counter(v for v in values if v is not None)
    total = sum(counter.values())

    if total == 0:
        return {}

    return {k: round(v / total * 100, 2) for k, v in counter.items()}


def compare_distributions(
    train: list[tuple[Path, dict]],
    test: list[tuple[Path, dict]],
) -> dict[str, dict]:
    """
    Train/Test 분포 비교

    Returns:
        축별 분포 비교 결과
    """
    comparison = {}

    for axis in STRATIFY_AXES:
        train_dist = calculate_distribution(train, axis)
        test_dist = calculate_distribution(test, axis)

        # 모든 키 합집합
        all_keys = set(train_dist.keys()) | set(test_dist.keys())

        # 차이 계산
        diffs = {}
        max_diff = 0.0
        for key in all_keys:
            train_val = train_dist.get(key, 0)
            test_val = test_dist.get(key, 0)
            diff = abs(train_val - test_val)
            diffs[key] = {
                "train": train_val,
                "test": test_val,
                "diff": round(diff, 2),
            }
            max_diff = max(max_diff, diff)

        comparison[axis] = {
            "distribution": diffs,
            "max_diff": round(max_diff, 2),
            "within_threshold": max_diff <= MAX_DISTRIBUTION_DIFF,
        }

    return comparison


def save_scenarios(
    scenarios: list[tuple[Path, dict]],
    target_dir: Path,
    dry_run: bool = False,
) -> int:
    """시나리오를 대상 디렉토리에 저장"""
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_path, scenario in scenarios:
        if not dry_run:
            dst_path = target_dir / src_path.name
            shutil.copy2(src_path, dst_path)
        count += 1

    return count


def generate_report(
    train: list[tuple[Path, dict]],
    test: list[tuple[Path, dict]],
    comparison: dict[str, dict],
) -> str:
    """분포 비교 리포트 생성"""
    lines = [
        "# Train/Test 분포 비교 리포트",
        "",
        f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 요약",
        f"- **Train**: {len(train)}개 ({len(train) / (len(train) + len(test)) * 100:.1f}%)",
        f"- **Test**: {len(test)}개 ({len(test) / (len(train) + len(test)) * 100:.1f}%)",
        f"- **검증 기준**: 분포 차이 {MAX_DISTRIBUTION_DIFF}% 이내",
        "",
        "## 축별 분포 비교",
        "",
    ]

    all_pass = True
    for axis, data in comparison.items():
        status = "✅ PASS" if data["within_threshold"] else "❌ FAIL"
        if not data["within_threshold"]:
            all_pass = False

        lines.append(f"### {axis}")
        lines.append(f"- 최대 차이: {data['max_diff']}% {status}")
        lines.append("")
        lines.append("| 값 | Train (%) | Test (%) | 차이 |")
        lines.append("|---|---|---|---|")

        for key, vals in sorted(data["distribution"].items()):
            diff_marker = "⚠️" if vals["diff"] > MAX_DISTRIBUTION_DIFF else ""
            lines.append(f"| {key} | {vals['train']} | {vals['test']} | {vals['diff']} {diff_marker} |")
        lines.append("")

    # 전체 결과
    lines.append("## 전체 검증 결과")
    if all_pass:
        lines.append("✅ **모든 축에서 분포 차이가 5% 이내입니다.**")
    else:
        failed_axes = [axis for axis, data in comparison.items() if not data["within_threshold"]]
        lines.append(f"⚠️ **다음 축에서 분포 차이가 5%를 초과합니다:** {', '.join(failed_axes)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train/Test 데이터셋 분리")
    parser.add_argument("--dry-run", action="store_true", help="실제 파일 생성 없이 테스트만 수행")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="랜덤 시드 (기본: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("Train/Test 데이터셋 분리")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/5] 시나리오 로드 중...")
    scenarios = load_all_scenarios()
    print(f"  총 {len(scenarios)}개 시나리오 로드 완료")

    # 2. 층화 분리
    print("\n[2/5] 층화 추출 중...")
    train, test = stratified_split(scenarios, seed=args.seed)
    print(f"  Train: {len(train)}개 ({len(train)/len(scenarios)*100:.1f}%)")
    print(f"  Test: {len(test)}개 ({len(test)/len(scenarios)*100:.1f}%)")

    # 3. 분포 비교
    print("\n[3/5] 분포 검증 중...")
    comparison = compare_distributions(train, test)

    all_pass = True
    for axis, data in comparison.items():
        status = "✅" if data["within_threshold"] else "❌"
        if not data["within_threshold"]:
            all_pass = False
        print(f"  {axis}: 최대 차이 {data['max_diff']}% {status}")

    # 4. 파일 저장
    print("\n[4/5] 파일 저장 중...")
    if args.dry_run:
        print("  (dry-run 모드: 파일 생성 건너뜀)")
    else:
        # 기존 디렉토리 삭제
        if TRAIN_DIR.exists():
            shutil.rmtree(TRAIN_DIR)
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)

        train_count = save_scenarios(train, TRAIN_DIR, args.dry_run)
        test_count = save_scenarios(test, TEST_DIR, args.dry_run)
        print(f"  Train: {train_count}개 -> {TRAIN_DIR}")
        print(f"  Test: {test_count}개 -> {TEST_DIR}")

    # 5. 메타데이터 및 리포트 저장
    print("\n[5/5] 메타데이터 및 리포트 저장 중...")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "test_ratio": TEST_RATIO,
        "total_scenarios": len(scenarios),
        "train_count": len(train),
        "test_count": len(test),
        "train_files": [p.name for p, _ in train],
        "test_files": [p.name for p, _ in test],
        "distribution_comparison": comparison,
        "all_axes_pass": all_pass,
        "max_allowed_diff_percent": MAX_DISTRIBUTION_DIFF,
    }

    if not args.dry_run:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  메타데이터: {METADATA_FILE}")

        # 리포트 생성
        report = generate_report(train, test, comparison)
        report_path = REPORT_DIR / "train_test_split_report.md"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  분포 리포트: {report_path}")
    else:
        print("  (dry-run 모드: 저장 건너뜀)")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"  분리 완료: Train {len(train)}개, Test {len(test)}개")
    print(f"  분포 검증: {'✅ 모든 축 PASS' if all_pass else '⚠️ 일부 축 FAIL'}")

    if not all_pass:
        print("\n  ⚠️ 분포 차이가 5%를 초과하는 축이 있습니다.")
        print("  층화 추출의 한계로 인한 것이며, 테스트 평가 시 고려가 필요합니다.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
