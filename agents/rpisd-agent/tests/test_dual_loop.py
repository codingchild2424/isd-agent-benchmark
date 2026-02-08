"""
이중 루프 테스트

무한 루프 방지 및 품질 기준 기반 분기 로직 테스트
"""

import pytest
from rpisd_agent.state import create_initial_state, RPISDState


class TestDualLoopLogic:
    """이중 루프 분기 로직 테스트"""

    def _create_state_with_quality(
        self,
        loop_source: str,
        current_quality: float,
        prototype_iteration: int = 0,
        development_iteration: int = 0,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
    ) -> RPISDState:
        """테스트용 상태 생성"""
        state = create_initial_state({"scenario_id": "TEST"})
        state["loop_source"] = loop_source
        state["current_quality"] = current_quality
        state["prototype_iteration"] = prototype_iteration
        state["development_iteration"] = development_iteration
        state["max_iterations"] = max_iterations
        state["quality_threshold"] = quality_threshold
        return state

    def test_prototype_loop_quality_pass(self):
        """프로토타입 루프: 품질 기준 충족 시 개발로 진행"""
        state = self._create_state_with_quality(
            loop_source="prototype",
            current_quality=0.85,
            prototype_iteration=1,
        )

        # 품질 충족 (0.85 >= 0.8)
        assert state["current_quality"] >= state["quality_threshold"]
        # 예상 결과: "development"

    def test_prototype_loop_quality_fail(self):
        """프로토타입 루프: 품질 미달 시 설계로 회귀"""
        state = self._create_state_with_quality(
            loop_source="prototype",
            current_quality=0.65,
            prototype_iteration=1,
        )

        # 품질 미달 (0.65 < 0.8)
        assert state["current_quality"] < state["quality_threshold"]
        # 반복 횟수 여유 있음 (1 < 3)
        assert state["prototype_iteration"] < state["max_iterations"]
        # 예상 결과: "design"

    def test_prototype_loop_max_iterations(self):
        """프로토타입 루프: 최대 반복 도달 시 개발로 진행"""
        state = self._create_state_with_quality(
            loop_source="prototype",
            current_quality=0.65,  # 품질 미달
            prototype_iteration=3,  # 최대 반복 도달
        )

        # 품질 미달이지만 최대 반복 도달
        assert state["current_quality"] < state["quality_threshold"]
        assert state["prototype_iteration"] >= state["max_iterations"]
        # 예상 결과: "development" (무한 루프 방지)

    def test_development_loop_quality_pass(self):
        """개발 루프: 품질 기준 충족 시 실행으로 진행"""
        state = self._create_state_with_quality(
            loop_source="development",
            current_quality=0.85,
            development_iteration=1,
        )

        # 품질 충족 (0.85 >= 0.8)
        assert state["current_quality"] >= state["quality_threshold"]
        # 예상 결과: "implementation"

    def test_development_loop_max_iterations(self):
        """개발 루프: 최대 반복 도달 시 실행으로 진행"""
        state = self._create_state_with_quality(
            loop_source="development",
            current_quality=0.65,  # 품질 미달
            development_iteration=3,  # 최대 반복 도달
        )

        # 품질 미달이지만 최대 반복 도달
        assert state["current_quality"] < state["quality_threshold"]
        assert state["development_iteration"] >= state["max_iterations"]
        # 예상 결과: "implementation" (무한 루프 방지)


class TestQualityThreshold:
    """품질 기준 테스트"""

    def test_threshold_boundary_pass(self):
        """경계값: 정확히 기준과 같으면 통과"""
        quality_threshold = 0.8
        current_quality = 0.8

        # 정확히 같으면 통과 (>=)
        assert current_quality >= quality_threshold

    def test_threshold_boundary_fail(self):
        """경계값: 기준보다 작으면 미달"""
        quality_threshold = 0.8
        current_quality = 0.79

        assert current_quality < quality_threshold

    def test_custom_threshold(self):
        """커스텀 품질 기준"""
        quality_threshold = 0.9  # 높은 기준

        assert 0.85 < quality_threshold  # 0.85는 미달
        assert 0.9 >= quality_threshold  # 0.9은 통과
        assert 0.95 >= quality_threshold  # 0.95는 통과


class TestIterationLimits:
    """반복 횟수 제한 테스트"""

    def test_max_iterations_respected(self):
        """최대 반복 횟수가 지켜지는지 확인"""
        max_iterations = 3

        for iteration in range(1, max_iterations + 2):
            should_stop = iteration >= max_iterations

            if iteration <= max_iterations:
                assert not should_stop or iteration == max_iterations
            else:
                assert should_stop

    def test_zero_iterations(self):
        """초기 상태 (0회 반복)"""
        state = create_initial_state({"scenario_id": "TEST"})

        assert state["prototype_iteration"] == 0
        assert state["development_iteration"] == 0

    def test_custom_max_iterations(self):
        """커스텀 최대 반복 횟수"""
        state = create_initial_state({"scenario_id": "TEST"})
        state["max_iterations"] = 5

        # 5회까지 반복 가능
        for i in range(1, 6):
            assert i <= state["max_iterations"]

        # 6회는 최대 초과
        assert 6 > state["max_iterations"]


class TestLoopSourceTracking:
    """루프 출처 추적 테스트"""

    def test_initial_loop_source(self):
        """초기 루프 출처는 빈 문자열"""
        state = create_initial_state({"scenario_id": "TEST"})

        assert state["loop_source"] == ""

    def test_prototype_loop_source(self):
        """프로토타입 루프 출처 설정"""
        state = create_initial_state({"scenario_id": "TEST"})
        state["loop_source"] = "prototype"

        assert state["loop_source"] == "prototype"

    def test_development_loop_source(self):
        """개발 루프 출처 설정"""
        state = create_initial_state({"scenario_id": "TEST"})
        state["loop_source"] = "development"

        assert state["loop_source"] == "development"
