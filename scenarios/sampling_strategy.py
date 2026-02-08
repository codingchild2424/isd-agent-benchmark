#!/usr/bin/env python3
"""
층화 샘플링 전략 모듈

IDLD 데이터셋의 불균형 축을 고려한 층화 샘플링을 제공합니다.
불균형 축의 희소 케이스를 오버샘플링하여 벤치마크 평가의 공정성을 확보합니다.

사용법:
    from sampling_strategy import StratifiedScenarioSampler

    sampler = StratifiedScenarioSampler()
    balanced_scenarios = sampler.sample(n_samples=100)
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# 설정
SCENARIOS_DIR = Path(__file__).parent
KOREAN_DIR = SCENARIOS_DIR / "idld_aligned"
ENGLISH_DIR = SCENARIOS_DIR / "idld_aligned_en"
AXIS_MAPPING_FILE = SCENARIOS_DIR / "axis_mapping_ko_en.json"

# 불균형 축 정의 (이슈 #92 기반)
IMBALANCED_AXES = {
    "difficulty": {
        "target_ratio": {"쉬움 - 단순한 구조, 제약조건 최소": 0.33,
                        "보통 - 일반적인 복잡도, 몇 가지 제약조건": 0.34,
                        "어려움 - 복잡한 요구사항, 다수의 제약조건": 0.33},
        "current_values": ["보통 - 일반적인 복잡도, 몇 가지 제약조건"]  # 현재 단일 값
    },
    "constraints.assessment_type": {
        "target_ratio": {"형성평가 중심": 0.33, "총괄평가 중심": 0.33, "프로젝트 기반 평가": 0.34},
        "min_ratio": 0.10  # 최소 10%
    },
    "context.learner_role": {
        "target_ratio": {"학생": 0.25, "직장인(사무/관리)": 0.25,
                        "전문직(의료/법률/기술)": 0.25, "예비 교사/교사": 0.25},
        "min_ratio": 0.05  # 최소 5%
    },
    "context.domain_expertise": {
        "target_ratio": {"초급": 0.33, "중급": 0.34, "고급": 0.33},
        "min_ratio": 0.10  # 최소 10%
    }
}


class StratifiedScenarioSampler:
    """불균형 축을 고려한 층화 샘플러"""

    def __init__(self, scenarios_dir: Path = KOREAN_DIR, seed: int | None = None):
        """
        Args:
            scenarios_dir: 시나리오 폴더 경로
            seed: 랜덤 시드 (재현성을 위해)
        """
        self.scenarios_dir = scenarios_dir
        self.scenarios: list[dict] = []
        self.scenario_paths: list[Path] = []
        self.stratified_index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

        if seed is not None:
            random.seed(seed)

        self._load_scenarios()
        self._build_stratified_index()

    def _load_scenarios(self) -> None:
        """모든 시나리오 로드"""
        for filepath in sorted(self.scenarios_dir.glob("*.json")):
            with open(filepath, "r", encoding="utf-8") as f:
                scenario = json.load(f)
                self.scenarios.append(scenario)
                self.scenario_paths.append(filepath)

    def _get_nested_value(self, scenario: dict, key: str) -> Any:
        """중첩 키로 값 추출 (예: 'context.learner_role')"""
        keys = key.split(".")
        value = scenario
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        return value

    def _build_stratified_index(self) -> None:
        """축별 인덱스 구축"""
        for idx, scenario in enumerate(self.scenarios):
            for axis in IMBALANCED_AXES.keys():
                value = self._get_nested_value(scenario, axis)
                if value:
                    self.stratified_index[axis][value].append(idx)

    def get_distribution(self) -> dict[str, dict[str, int]]:
        """현재 데이터 분포 반환"""
        distribution = {}
        for axis, values in self.stratified_index.items():
            distribution[axis] = {v: len(indices) for v, indices in values.items()}
        return distribution

    def sample_balanced(self, n_samples: int, strategy: str = "oversample") -> list[dict]:
        """
        균형 잡힌 샘플 추출

        Args:
            n_samples: 추출할 샘플 수
            strategy: 샘플링 전략
                - "oversample": 희소 케이스 오버샘플링 (기본)
                - "undersample": 다수 케이스 언더샘플링
                - "proportional": 목표 비율에 비례

        Returns:
            균형 잡힌 시나리오 리스트
        """
        if strategy == "oversample":
            return self._oversample(n_samples)
        elif strategy == "undersample":
            return self._undersample(n_samples)
        elif strategy == "proportional":
            return self._proportional_sample(n_samples)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _oversample(self, n_samples: int) -> list[dict]:
        """
        희소 케이스 오버샘플링

        불균형 축의 희소 값을 가진 시나리오를 우선 선택하고,
        부족한 경우 중복 허용하여 목표 비율 달성
        """
        selected_indices = set()
        weighted_indices: list[int] = []

        # 1단계: 희소 케이스 우선 수집
        for axis, config in IMBALANCED_AXES.items():
            axis_values = self.stratified_index[axis]
            total = sum(len(indices) for indices in axis_values.values())

            for value, indices in axis_values.items():
                current_ratio = len(indices) / total if total > 0 else 0
                target_ratio = config.get("target_ratio", {}).get(value, 1.0 / len(axis_values))

                # 현재 비율이 목표보다 낮으면 가중치 부여
                if current_ratio < target_ratio:
                    weight = int((target_ratio / max(current_ratio, 0.01)) * 2)
                    weighted_indices.extend(indices * weight)
                else:
                    weighted_indices.extend(indices)

        # 2단계: 가중치 기반 랜덤 샘플링
        if len(weighted_indices) < n_samples:
            # 전체 시나리오에서 추가 샘플링
            all_indices = list(range(len(self.scenarios)))
            weighted_indices.extend(all_indices * (n_samples // len(all_indices) + 1))

        sampled_indices = random.sample(weighted_indices, min(n_samples, len(weighted_indices)))

        return [self.scenarios[idx] for idx in sampled_indices]

    def _undersample(self, n_samples: int) -> list[dict]:
        """
        다수 케이스 언더샘플링

        불균형 축의 다수 값을 가진 시나리오를 제한하여 선택
        """
        # 각 축별 균형 수량 계산
        samples_per_value = n_samples // 12  # 대략적인 축/값 조합 수

        selected_indices = set()

        for axis, config in IMBALANCED_AXES.items():
            axis_values = self.stratified_index[axis]

            for value, indices in axis_values.items():
                # 최대 samples_per_value개만 선택
                sample_count = min(len(indices), samples_per_value)
                sampled = random.sample(indices, sample_count)
                selected_indices.update(sampled)

        # 목표 수량 조정
        if len(selected_indices) > n_samples:
            selected_indices = set(random.sample(list(selected_indices), n_samples))
        elif len(selected_indices) < n_samples:
            remaining = n_samples - len(selected_indices)
            all_indices = set(range(len(self.scenarios))) - selected_indices
            additional = random.sample(list(all_indices), min(remaining, len(all_indices)))
            selected_indices.update(additional)

        return [self.scenarios[idx] for idx in selected_indices]

    def _proportional_sample(self, n_samples: int) -> list[dict]:
        """
        목표 비율에 비례한 샘플링

        각 불균형 축의 목표 비율에 맞춰 시나리오 선택
        """
        # 주요 축 선택 (domain_expertise 기준)
        axis = "context.domain_expertise"
        config = IMBALANCED_AXES[axis]
        target_ratios = config.get("target_ratio", {})

        selected = []
        for value, ratio in target_ratios.items():
            indices = self.stratified_index[axis].get(value, [])
            target_count = int(n_samples * ratio)

            if len(indices) >= target_count:
                sampled = random.sample(indices, target_count)
            else:
                # 부족하면 중복 허용
                sampled = random.choices(indices, k=target_count) if indices else []

            selected.extend([self.scenarios[idx] for idx in sampled])

        return selected[:n_samples]

    def sample_with_paths(self, n_samples: int, strategy: str = "oversample") -> list[tuple[Path, dict]]:
        """
        파일 경로와 함께 샘플 추출

        Returns:
            (파일경로, 시나리오) 튜플 리스트
        """
        scenarios = self.sample_balanced(n_samples, strategy)

        # 시나리오에서 경로 찾기
        result = []
        scenario_to_path = {s["scenario_id"]: p for s, p in zip(self.scenarios, self.scenario_paths)}

        for scenario in scenarios:
            path = scenario_to_path.get(scenario["scenario_id"])
            if path:
                result.append((path, scenario))

        return result

    def get_balance_report(self, scenarios: list[dict]) -> dict:
        """
        샘플의 균형 상태 리포트

        Args:
            scenarios: 검사할 시나리오 리스트

        Returns:
            축별 분포 및 균형 상태
        """
        report = {}

        for axis in IMBALANCED_AXES.keys():
            counter = Counter()
            for scenario in scenarios:
                value = self._get_nested_value(scenario, axis)
                if value:
                    counter[value] += 1

            total = sum(counter.values())
            distribution = {}
            for value, count in counter.most_common():
                ratio = count / total if total > 0 else 0
                distribution[value] = {"count": count, "ratio": round(ratio * 100, 2)}

            # 불균형 판정 (상위 값이 50% 초과면 불균형)
            top_ratio = list(distribution.values())[0]["ratio"] if distribution else 0
            is_balanced = top_ratio <= 50

            report[axis] = {
                "distribution": distribution,
                "is_balanced": is_balanced,
                "top_ratio": top_ratio
            }

        return report


def main():
    """테스트 및 데모"""
    print("=" * 60)
    print("층화 샘플링 전략 테스트")
    print("=" * 60)

    sampler = StratifiedScenarioSampler(seed=42)

    print(f"\n총 시나리오 수: {len(sampler.scenarios)}")

    print("\n[현재 분포]")
    distribution = sampler.get_distribution()
    for axis, values in distribution.items():
        print(f"\n{axis}:")
        total = sum(values.values())
        for value, count in sorted(values.items(), key=lambda x: -x[1]):
            ratio = count / total * 100 if total > 0 else 0
            print(f"  {value}: {count} ({ratio:.1f}%)")

    # 균형 샘플링 테스트
    print("\n" + "=" * 60)
    print("균형 샘플링 테스트 (100개)")
    print("=" * 60)

    for strategy in ["oversample", "undersample", "proportional"]:
        print(f"\n[전략: {strategy}]")
        sampled = sampler.sample_balanced(100, strategy=strategy)
        report = sampler.get_balance_report(sampled)

        for axis, info in report.items():
            status = "✅ 균형" if info["is_balanced"] else "⚠️ 불균형"
            print(f"  {axis}: {status} (최대 {info['top_ratio']}%)")


if __name__ == "__main__":
    main()
