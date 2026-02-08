#!/usr/bin/env python3
# 워닝 필터 (Python 3.14 + Pydantic V1 호환성 경고 숨김)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

"""
ISD Agent Benchmark 통합 테스트 스크립트

3종 Agent(EduPlanner, Baseline-SolarPro2, ReAct-ISD)를
여러 시나리오에 대해 실행하고 비교 평가합니다.
Upstage Solar Pro2 API를 사용합니다.
"""

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Dict

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

# Agent 모듈 경로 추가
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "baseline" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "eduplanner" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "react-isd" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "addie-agent" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "dick-carey-agent" / "src"))
sys.path.insert(0, str(_SCRIPT_DIR / "agents" / "rpisd-agent" / "src"))


class SolarKeyRotator:
    """Solar Pro 3용 라운드 로빈 API 키 로테이터"""

    def __init__(self):
        self._lock = threading.Lock()
        self._index = 0

        # 사용 가능한 키 목록 구성
        self._keys = []

        # 1. OpenRouter (upstage/solar-pro-3:free)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            self._keys.append({
                "provider": "openrouter",
                "model": "upstage/solar-pro-3:free",
                "api_key": openrouter_key,
                "base_url": "https://openrouter.ai/api/v1",
            })

        # 2. Upstage Direct (UPSTAGE_API_KEY)
        upstage_key = os.getenv("UPSTAGE_API_KEY")
        if upstage_key:
            self._keys.append({
                "provider": "upstage",
                "model": "solar-pro3",
                "api_key": upstage_key,
                "base_url": "https://api.upstage.ai/v1/solar",
            })

        # 3. Upstage Direct 2 (UPSTAGE_API_KEY2)
        upstage_key2 = os.getenv("UPSTAGE_API_KEY2")
        if upstage_key2:
            self._keys.append({
                "provider": "upstage2",
                "model": "solar-pro3",
                "api_key": upstage_key2,
                "base_url": "https://api.upstage.ai/v1/solar",
            })

        if not self._keys:
            raise ValueError("No API keys found for Solar Pro 3")

        print(f"  [SolarKeyRotator] {len(self._keys)}개 키 로드됨: {[k['provider'] for k in self._keys]}")

    def get_next(self) -> dict:
        """라운드 로빈으로 다음 키 반환"""
        with self._lock:
            key_info = self._keys[self._index]
            self._index = (self._index + 1) % len(self._keys)
            return key_info


# 전역 Solar 키 로테이터 (Solar 모델 사용 시에만 초기화)
_solar_key_rotator: Optional[SolarKeyRotator] = None


def get_solar_key_rotator() -> SolarKeyRotator:
    """Solar 키 로테이터 싱글톤 반환"""
    global _solar_key_rotator
    if _solar_key_rotator is None:
        _solar_key_rotator = SolarKeyRotator()
    return _solar_key_rotator


class BenchmarkProgressLogger:
    """벤치마크 진행 상황을 친절하게 출력하는 로거"""

    def __init__(self, total_scenarios: int, total_agents: int, log_file: Optional[Path] = None):
        self.total_scenarios = total_scenarios
        self.total_agents = total_agents
        self.total_tasks = total_scenarios * total_agents  # 총 작업 수

        self.completed_scenarios = 0
        self.completed_tasks = 0
        self.start_time = time.time()
        self.scenario_times = []  # 시나리오별 소요시간 기록

        self.log_file = log_file
        self.lock = threading.Lock()

        # 로그 파일 초기화
        if self.log_file:
            self._write_log_header()

    def _write_log_header(self):
        """로그 파일 헤더 작성"""
        header = f"""
================================================================================
  ISD Agent Benchmark 실행 로그
  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

총 시나리오: {self.total_scenarios}개
총 에이전트: {self.total_agents}개
총 작업 수: {self.total_tasks}개 (시나리오 × 에이전트)

================================================================================
"""
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(header)
        print(header)

    def _estimate_remaining_time(self) -> str:
        """남은 시간 예측"""
        if not self.scenario_times:
            return "계산 중..."

        avg_time = sum(self.scenario_times) / len(self.scenario_times)
        remaining_scenarios = self.total_scenarios - self.completed_scenarios
        remaining_seconds = avg_time * remaining_scenarios

        if remaining_seconds < 60:
            return f"{int(remaining_seconds)}초"
        elif remaining_seconds < 3600:
            return f"{int(remaining_seconds / 60)}분 {int(remaining_seconds % 60)}초"
        else:
            hours = int(remaining_seconds / 3600)
            minutes = int((remaining_seconds % 3600) / 60)
            return f"{hours}시간 {minutes}분"

    def _estimate_completion_time(self) -> str:
        """예상 완료 시간"""
        if not self.scenario_times:
            return "계산 중..."

        avg_time = sum(self.scenario_times) / len(self.scenario_times)
        remaining_scenarios = self.total_scenarios - self.completed_scenarios
        remaining_seconds = avg_time * remaining_scenarios

        completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
        return completion_time.strftime('%H:%M:%S')

    def _get_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """진행 바 생성"""
        if total == 0:
            return "░" * width

        filled = int(width * current / total)
        empty = width - filled
        percentage = (current / total) * 100

        return f"{'█' * filled}{'░' * empty} {percentage:5.1f}%"

    def _format_elapsed_time(self) -> str:
        """경과 시간 포맷"""
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{int(elapsed)}초"
        elif elapsed < 3600:
            return f"{int(elapsed / 60)}분 {int(elapsed % 60)}초"
        else:
            hours = int(elapsed / 3600)
            minutes = int((elapsed % 3600) / 60)
            return f"{hours}시간 {minutes}분"

    def log_scenario_start(self, scenario_id: str, scenario_index: int):
        """시나리오 시작 로그"""
        with self.lock:
            progress_bar = self._get_progress_bar(scenario_index, self.total_scenarios)
            remaining = self._estimate_remaining_time()
            completion = self._estimate_completion_time()

            log_msg = f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 📊 진행 현황: [{progress_bar}]
│
│ 🔄 현재 작업: [{scenario_index + 1}/{self.total_scenarios}] {scenario_id}
│ ⏱️  경과 시간: {self._format_elapsed_time()}
│ ⏳ 예상 남은 시간: {remaining}
│ 🏁 예상 완료 시간: {completion}
│
│ 남은 시나리오: {self.total_scenarios - scenario_index - 1}개
└─────────────────────────────────────────────────────────────────────────────┘
"""
            print(log_msg)
            if self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] 시작: {scenario_id} ({scenario_index + 1}/{self.total_scenarios})\n")

    def log_agent_progress(self, scenario_id: str, agent_id: str, agent_index: int,
                           total_agents: int, status: str):
        """에이전트 진행 로그"""
        with self.lock:
            status_icon = "✅" if status == "success" else "❌" if status == "failed" else "🔄"
            log_msg = f"    {status_icon} [{agent_index + 1}/{total_agents}] {agent_id}: {status}"
            print(log_msg)

            if self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"  - {agent_id}: {status}\n")

    def log_scenario_complete(self, scenario_id: str, elapsed_seconds: float, success_count: int):
        """시나리오 완료 로그"""
        with self.lock:
            self.completed_scenarios += 1
            self.scenario_times.append(elapsed_seconds)

            log_msg = f"""
    ────────────────────────────────────────────────────────────
    ✅ 완료: {scenario_id}
    ⏱️  소요 시간: {elapsed_seconds:.1f}초
    📈 성공한 에이전트: {success_count}/{self.total_agents}
    ────────────────────────────────────────────────────────────
"""
            print(log_msg)

            if self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 완료: {scenario_id} ({elapsed_seconds:.1f}초, 성공: {success_count})\n")

    def log_final_summary(self, results: dict):
        """최종 요약 로그"""
        total_elapsed = time.time() - self.start_time

        # 통계 계산
        total_success = 0
        total_failed = 0
        for variant_results in results.get("scenarios", {}).values():
            for scenario_result in variant_results.values():
                for agent_result in scenario_result.get("agents", {}).values():
                    if agent_result.get("success"):
                        total_success += 1
                    else:
                        total_failed += 1

        summary = f"""

================================================================================
  🏁 벤치마크 완료!
================================================================================

📊 최종 결과
────────────────────────────────────────────────────────────────────────────────
  총 시나리오:     {self.completed_scenarios}개
  총 작업 수:      {total_success + total_failed}개
  성공:           {total_success}개 ✅
  실패:           {total_failed}개 ❌
  성공률:         {(total_success / (total_success + total_failed) * 100) if (total_success + total_failed) > 0 else 0:.1f}%

⏱️  총 소요 시간: {self._format_elapsed_time()}
📁 결과 저장 위치: {results.get('output_dir', 'N/A')}

================================================================================
"""
        print(summary)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(summary)

# .env 파일에서 환경변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv가 없으면 건너뜀


# 프로젝트 루트 경로 (isd-agent-bench-en 디렉토리)
PROJECT_ROOT = Path(__file__).parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
RESULTS_DIR = PROJECT_ROOT / "results"
VENV_BIN = PROJECT_ROOT / ".venv" / "bin"

# 환경변수 설정 (Agent 실행용)
def get_env_with_venv():
    """venv bin 경로가 포함된 환경변수 반환"""
    env = os.environ.copy()
    venv_path = str(VENV_BIN)
    current_path = env.get('PATH', '')
    env['PATH'] = f"{venv_path}:{current_path}"
    return env


def get_all_scenarios(
    use_stratified_sampling: bool = False,
    n_samples: int | None = None,
    sampling_strategy: str = "oversample",
    dataset: str | None = None,
) -> dict[str, list[Path]]:
    """
    모든 시나리오 파일을 variant별로 수집 (IDLD 데이터셋 구조)

    데이터셋 구조 관계:
    - dataset=None (기본): 기존 variant 디렉토리 사용 (idld_aligned, context_variant)
    - dataset="train": 학습용 데이터셋 (scenarios/train/) - idld_aligned + context_variant에서 분리
    - dataset="test": 평가용 데이터셋 (scenarios/test/) - Hold-out 평가용, 5% 비율

    Args:
        use_stratified_sampling: 층화 샘플링 사용 여부 (불균형 축 보정)
        n_samples: 샘플링 시 추출할 시나리오 수 (None이면 전체)
        sampling_strategy: 샘플링 전략 ("oversample", "undersample", "proportional")
        dataset: 데이터셋 선택 ("train", "test", None=variant 모드)

    Returns:
        variant/dataset별 시나리오 파일 경로 딕셔너리
    """
    # dataset 모드: train/test 디렉토리에서 직접 로드
    if dataset in ("train", "test"):
        dataset_dir = SCENARIOS_DIR / dataset
        if not dataset_dir.exists():
            print(f"[경고] {dataset} 디렉토리가 존재하지 않습니다: {dataset_dir}")
            return {dataset: []}

        scenario_files = sorted(dataset_dir.glob("*.json"))
        print(f"  [{dataset.upper()}] {len(scenario_files)}개 시나리오 로드됨")

        # 층화 샘플링 지원 (train 데이터셋에서만 의미 있음)
        if use_stratified_sampling and n_samples and dataset == "train":
            try:
                from scenarios.sampling_strategy import StratifiedScenarioSampler
                sampler = StratifiedScenarioSampler(scenarios_dir=dataset_dir)
                sampled = sampler.sample_with_paths(n_samples, strategy=sampling_strategy)
                scenario_files = [path for path, _ in sampled]
                print(f"  [층화 샘플링] {dataset}: {len(scenario_files)}개 선택 (전략: {sampling_strategy})")
            except ImportError:
                pass  # 샘플링 모듈 없으면 전체 사용

        return {dataset: scenario_files}

    # 기존 variant 모드: idld_aligned, context_variant 디렉토리 사용
    scenarios = {"idld_aligned": [], "context_variant": []}

    for variant in scenarios.keys():
        variant_dir = SCENARIOS_DIR / variant
        if variant_dir.exists():
            if use_stratified_sampling and variant == "idld_aligned" and n_samples:
                # 층화 샘플링 적용 (불균형 축 보정)
                try:
                    from scenarios.sampling_strategy import StratifiedScenarioSampler
                    sampler = StratifiedScenarioSampler(scenarios_dir=variant_dir)
                    sampled = sampler.sample_with_paths(n_samples, strategy=sampling_strategy)
                    scenarios[variant] = [path for path, _ in sampled]
                    print(f"  [층화 샘플링] {variant}: {len(scenarios[variant])}개 선택 (전략: {sampling_strategy})")
                except ImportError:
                    # 샘플링 모듈 없으면 기본 동작
                    for scenario_file in sorted(variant_dir.glob("*.json")):
                        scenarios[variant].append(scenario_file)
            else:
                for scenario_file in sorted(variant_dir.glob("*.json")):
                    scenarios[variant].append(scenario_file)

    return scenarios


def install_agents() -> bool:
    """모든 Agent 패키지 설치"""
    print("\n" + "=" * 60)
    print("Agent 패키지 설치")
    print("=" * 60)

    agents = [
        PROJECT_ROOT / "agents" / "eduplanner",
        PROJECT_ROOT / "agents" / "baseline",
        PROJECT_ROOT / "agents" / "react-isd",
        PROJECT_ROOT / "agents" / "addie-agent",
        PROJECT_ROOT / "agents" / "dick-carey-agent",
        PROJECT_ROOT / "agents" / "rpisd-agent",
        PROJECT_ROOT / "evaluator",
    ]

    for agent_path in agents:
        print(f"\n설치 중: {agent_path.name}")
        result = subprocess.run(
            ["pip", "install", "-e", str(agent_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  오류: {result.stderr}")
            return False
        print(f"  완료")

    return True


def check_agents_installed() -> dict[str, bool]:
    """Agent 모듈 import 가능 여부 확인"""
    agents = {
        "eduplanner": False,
        "baseline": False,
        "react-isd": False,
        "addie-agent": False,
        "dick-carey-agent": False,
        "rpisd-agent": False,
    }

    # 모듈 import 테스트
    try:
        from baseline.generator import BaselineGenerator
        agents["baseline"] = True
    except ImportError:
        pass

    try:
        from eduplanner.agents import EduPlannerAgent
        agents["eduplanner"] = True
    except ImportError:
        pass

    try:
        from react_isd.agent import ReActISDAgent
        agents["react-isd"] = True
    except ImportError:
        pass

    try:
        from addie_agent.agent import ADDIEAgent
        agents["addie-agent"] = True
    except ImportError:
        pass

    try:
        from dick_carey_agent.agent import DickCareyAgent
        agents["dick-carey-agent"] = True
    except ImportError:
        pass

    try:
        from rpisd_agent.agent import RPISDAgent
        agents["rpisd-agent"] = True
    except ImportError:
        pass

    return agents


def _get_model_config() -> tuple[str, str, Optional[str]]:
    """모델 설정 반환. Solar Pro 3이면 키 로테이션 적용.

    Returns:
        (provider, model, api_key) - api_key는 Solar 로테이션 시에만 설정
    """
    provider = os.getenv("MODEL_PROVIDER", "openrouter")
    model = os.getenv("MODEL_NAME", "anthropic/claude-opus-4.5")

    # Solar Pro 3 모델 감지 (키 로테이션 적용)
    is_solar = "solar" in model.lower()
    if is_solar:
        rotator = get_solar_key_rotator()
        key_info = rotator.get_next()
        # 환경변수 임시 설정 (에이전트들이 읽을 수 있도록)
        os.environ["_ROTATED_API_KEY"] = key_info["api_key"]
        os.environ["_ROTATED_BASE_URL"] = key_info["base_url"]
        return key_info["provider"], key_info["model"], key_info["api_key"]

    return provider, model, None


def _get_agent_runner(agent_id: str):
    """에이전트 ID에 해당하는 실행 함수 반환 (모듈 기반)"""

    if agent_id == "baseline":
        from baseline.generator import BaselineGenerator
        def run_baseline(scenario: dict) -> dict:
            provider, model, api_key = _get_model_config()
            gen = BaselineGenerator(model=model, provider=provider, api_key=api_key)
            return gen.generate(scenario)
        return run_baseline

    elif agent_id == "eduplanner":
        from eduplanner.agents import EduPlannerAgent
        from eduplanner.agents.base import AgentConfig
        from eduplanner.models.schemas import ScenarioInput
        def run_eduplanner(scenario: dict) -> dict:
            provider, model, api_key = _get_model_config()
            config = AgentConfig(model=model, provider=provider)
            agent = EduPlannerAgent(config=config, max_iterations=3, target_score=90.0)
            scenario_input = ScenarioInput(**scenario)
            result = agent.run(scenario_input)
            return {
                "addie_output": result.addie_output.to_standard_dict(),
                "trajectory": result.trajectory.model_dump(),
                "metadata": result.metadata.model_dump(),
            }
        return run_eduplanner

    elif agent_id == "react-isd":
        from react_isd.agent import ReActISDAgent
        def run_react(scenario: dict) -> dict:
            provider, model, api_key = _get_model_config()
            agent = ReActISDAgent(model=model, provider=provider, api_key=api_key)
            return agent.run(scenario)
        return run_react

    elif agent_id == "addie-agent":
        from addie_agent.agent import ADDIEAgent
        def run_addie(scenario: dict) -> dict:
            provider, model, api_key = _get_model_config()
            agent = ADDIEAgent(model=model)
            return agent.run(scenario)
        return run_addie

    elif agent_id == "dick-carey-agent":
        from dick_carey_agent.agent import DickCareyAgent
        def run_dickcarey(scenario: dict) -> dict:
            provider, model, api_key = _get_model_config()
            agent = DickCareyAgent(model=model)
            return agent.run(scenario)
        return run_dickcarey

    elif agent_id == "rpisd-agent":
        from rpisd_agent.agent import RPISDAgent
        def run_rpisd(scenario: dict) -> dict:
            provider, model, api_key = _get_model_config()
            agent = RPISDAgent(model=model)
            return agent.run(scenario)
        return run_rpisd

    else:
        raise ValueError(f"Unknown agent: {agent_id}")


def _run_agent_task(
    agent_id: str,
    scenario_path: Path,
    output_dir: Path,
    semaphore: Optional[threading.Semaphore] = None,
) -> tuple[str, dict]:
    """개별 Agent 실행 태스크 (모듈 기반, 병렬 실행용)"""
    if semaphore:
        semaphore.acquire()

    try:
        output_path = output_dir / f"{agent_id}_output.json"
        trajectory_path = output_dir / f"{agent_id}_trajectory.json"
        log_path = output_dir / f"{agent_id}_log.txt"

        # 출력 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 시나리오 로드
            with open(scenario_path, "r", encoding="utf-8") as f:
                scenario = json.load(f)

            # 에이전트 실행 (모듈 기반)
            start_time = time.time()
            runner = _get_agent_runner(agent_id)
            result = runner(scenario)
            elapsed = time.time() - start_time

            # ADDIE 출력 저장
            addie_output = result.get("addie_output", result)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(addie_output, f, ensure_ascii=False, indent=2, default=str)

            # Trajectory 저장
            trajectory_data = {
                "scenario_id": scenario.get("scenario_id", "unknown"),
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "trajectory": result.get("trajectory", {}),
                "metadata": result.get("metadata", {"execution_time_seconds": elapsed}),
            }
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, ensure_ascii=False, indent=2, default=str)

            # 로그 저장
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== {agent_id} 실행 로그 ===\n")
                f.write(f"Scenario: {scenario_path}\n")
                f.write(f"Elapsed: {elapsed:.2f}s\n")
                f.write(f"Status: SUCCESS\n")

            return agent_id, {
                "success": True,
                "output_path": str(output_path),
                "trajectory_path": str(trajectory_path),
                "log_path": str(log_path),
            }

        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== {agent_id} 실행 로그 ===\n")
                f.write(f"Scenario: {scenario_path}\n")
                f.write(f"Status: FAILED\n")
                f.write(f"Error: {error_msg}\n\n")
                f.write("=== Traceback ===\n")
                f.write(tb)

            return agent_id, {
                "success": False,
                "error": error_msg[:500],
                "log_path": str(log_path),
            }

    finally:
        if semaphore:
            semaphore.release()


def run_single_benchmark(
    scenario_path: Path,
    output_dir: Path,
    agents: Optional[list[str]] = None,
    verbose: bool = False,
    parallel: bool = False,
    max_workers: int = 3,
    multi_judge: bool = True,
) -> dict:
    """단일 시나리오에 대해 벤치마크 실행

    Args:
        scenario_path: 시나리오 파일 경로
        output_dir: 출력 디렉토리
        agents: 실행할 Agent 목록
        verbose: 상세 출력 여부
        parallel: Agent 병렬 실행 여부
        max_workers: 최대 동시 실행 수
    """
    agents = agents or ["eduplanner", "baseline", "react-isd", "addie-agent", "dick-carey-agent", "rpisd-agent"]

    print(f"\n시나리오: {scenario_path.name}")
    print("-" * 40)

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "scenario": scenario_path.name,
        "agents": {},
        "timestamp": datetime.now().isoformat(),
    }

    if parallel and len(agents) > 1:
        # 병렬 실행: ThreadPoolExecutor + Semaphore
        effective_workers = max_workers
        print(f"  [병렬 모드] {len(agents)}개 Agent 동시 실행 (max_workers={effective_workers})")
        semaphore = threading.Semaphore(effective_workers)

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {}
            for idx, agent_id in enumerate(agents):
                future = executor.submit(
                    _run_agent_task, agent_id, scenario_path, output_dir, semaphore
                )
                futures[future] = agent_id
                # Rate limit 대응: Agent 제출 간 딜레이
                time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    _, agent_result = future.result()
                    status = "완료" if agent_result["success"] else "실패"
                    print(f"  {agent_id}: {status}")
                    results["agents"][agent_id] = agent_result

                    if verbose and not agent_result["success"]:
                        print(f"    stderr: {agent_result.get('stderr', '')[:200]}")
                except Exception as e:
                    print(f"  {agent_id}: 예외 발생")
                    results["agents"][agent_id] = {
                        "success": False,
                        "error": str(e),
                    }
    else:
        # 순차 실행 (기존 방식)
        for agent_id in agents:
            print(f"  {agent_id} 실행 중...", end=" ", flush=True)
            _, agent_result = _run_agent_task(agent_id, scenario_path, output_dir)

            status = "완료" if agent_result["success"] else "실패"
            print(status)
            results["agents"][agent_id] = agent_result

            if verbose and not agent_result["success"]:
                print(f"    stderr: {agent_result.get('stderr', '')[:200]}")

            # Rate limit 대응: Agent 간 딜레이
            time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

    # 평가 실행
    successful_agents = [a for a, r in results["agents"].items() if r.get("success")]

    if len(successful_agents) >= 2:
        print(f"  Evaluating...", end=" ", flush=True)

        cmd = [
            "isd-evaluator",
            "compare",
            "--scenario", str(scenario_path),
            "--output-dir", str(output_dir),
            "--agents", ",".join(successful_agents),
        ]

        # Use multi-judge (5 LLMs) or single-judge
        if multi_judge:
            cmd.append("--multi-judge")
        else:
            cmd.extend(["--single-judge", "--use-llm"])

        if verbose:
            cmd.append("--verbose")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=get_env_with_venv(),
            # 타임아웃 없음 - LLM 평가는 시간이 오래 걸릴 수 있음
        )

        if result.returncode == 0:
            print("완료")

            # 평가 결과 로드
            report_path = output_dir / "comparison_report.json"
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    results["evaluation"] = json.load(f)
        else:
            print("실패")
            results["evaluation_error"] = result.stderr[:500] if result.stderr else "Unknown error"
    else:
        print(f"  평가 생략 (성공한 Agent: {len(successful_agents)}개)")
        results["evaluation_skipped"] = True

    return results


def _run_scenario_task(
    scenario_path: Path,
    output_dir: Path,
    agents: Optional[list[str]],
    verbose: bool,
    parallel: bool,
    max_workers: int,
    multi_judge: bool = True,
    semaphore: Optional[threading.Semaphore] = None,
) -> tuple[str, dict]:
    """Scenario execution task (for scenario-level parallelization)"""
    if semaphore:
        semaphore.acquire()

    try:
        scenario_id = scenario_path.stem
        result = run_single_benchmark(
            scenario_path=scenario_path,
            output_dir=output_dir,
            agents=agents,
            verbose=verbose,
            parallel=parallel,
            max_workers=max_workers,
            multi_judge=multi_judge,
        )
        return scenario_id, result
    finally:
        if semaphore:
            semaphore.release()


def run_full_benchmark(
    variants: Optional[list[str]] = None,
    agents: Optional[list[str]] = None,
    verbose: bool = False,
    parallel: bool = True,
    max_workers: int = 6,
    scenario_parallel: bool = True,
    scenario_max_workers: int = 8,
    dataset: Optional[str] = None,
    multi_judge: bool = True,
) -> dict:
    """전체 벤치마크 실행 (IDLD 데이터셋 구조)

    Args:
        variants: 실행할 variant 목록 (dataset=None일 때만 사용)
        agents: 실행할 Agent 목록
        verbose: 상세 출력 여부
        parallel: Agent 레벨 병렬 실행 여부 (기본: True)
        max_workers: Agent 동시 실행 수 (기본: 6)
        scenario_parallel: 시나리오 레벨 병렬 실행 여부 (기본: True)
        scenario_max_workers: 시나리오 동시 실행 수 (기본: 8)
        dataset: 데이터셋 선택 ("train", "test", None=variant 모드)
    """
    # dataset 모드일 때는 variants 무시
    if dataset:
        variants = [dataset]  # dataset을 variant처럼 처리
    else:
        variants = variants or ["idld_aligned", "context_variant"]

    agents = agents or ["eduplanner", "baseline", "react-isd", "addie-agent", "dick-carey-agent", "rpisd-agent"]

    # 시나리오 수집 (먼저 수집하여 총 개수 파악)
    all_scenarios = get_all_scenarios(dataset=dataset)
    total_scenarios = sum(len(s) for s in all_scenarios.values())

    # 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 모델 이름에서 디렉토리용 safe name 생성 (예: anthropic/claude-opus-4.5 -> claude-opus-4.5)
    model_name = os.getenv("MODEL_NAME", "default")
    model_safe_name = model_name.split("/")[-1].replace(":", "-")  # Remove provider prefix and replace colons

    # dataset 모드일 때는 디렉토리 이름에 반영 (예: test_benchmark_claude-opus-4.5_20260122_...)
    if dataset:
        run_dir = RESULTS_DIR / f"{dataset}_benchmark_{model_safe_name}_{timestamp}"
    else:
        run_dir = RESULTS_DIR / f"benchmark_{model_safe_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 로그 파일 경로
    log_file = run_dir / "benchmark_progress.log"

    # 로거 초기화
    logger = BenchmarkProgressLogger(
        total_scenarios=total_scenarios,
        total_agents=len(agents),
        log_file=log_file
    )

    print(f"\n{'=' * 80}")
    print(f"  🚀 ISD Agent Benchmark 실행")
    print(f"{'=' * 80}")
    print(f"  🤖 Model: {model_name} (provider: {os.getenv('MODEL_PROVIDER', 'openrouter')})")

    if dataset:
        print(f"  📂 데이터셋 모드: {dataset.upper()}")
    else:
        print(f"  📂 Variant 모드: {', '.join(variants)}")

    print(f"  Agents ({len(agents)}): {', '.join(agents)}")
    print(f"  Total scenarios: {total_scenarios}")
    print(f"  Parallel mode: {scenario_max_workers} scenarios x {max_workers} agents concurrently")
    if multi_judge:
        print(f"  Evaluation: Multi-Judge (2 LLMs, 경량/빠름)")
        print(f"    - openai/gpt-4o-mini (OPENROUTER_API_KEY)")
        print(f"    - google/gemini-2.5-flash-lite (OPENROUTER_API_KEY)")
    else:
        print(f"  Evaluation: Single-Judge")
    print(f"  Output: {run_dir}")
    print(f"  Log file: {log_file}")
    print(f"{'=' * 80}\n")

    results = {
        "timestamp": timestamp,
        "output_dir": str(run_dir),
        "model": {
            "name": model_name,
            "provider": os.getenv("MODEL_PROVIDER", "openrouter"),
        },
        "config": {
            "variants": variants,
            "dataset": dataset,
            "agents": agents,
            "parallel": parallel,
            "max_workers": max_workers,
            "scenario_parallel": scenario_parallel,
            "scenario_max_workers": scenario_max_workers,
            "multi_judge": multi_judge,
            "judge_models": [
                "openai/gpt-4o-mini",
                "google/gemini-2.5-flash-lite",
            ] if multi_judge else None,
        },
        "scenarios": {},
    }

    scenario_index = 0

    # 각 variant별 시나리오 실행
    for variant in variants:
        scenarios = all_scenarios.get(variant, [])
        if not scenarios:
            print(f"\n[{variant}] 시나리오 없음")
            continue

        print(f"\n{'─' * 80}")
        print(f"  📁 [{variant.upper()}] {len(scenarios)}개 시나리오 시작")
        print(f"{'─' * 80}")

        results["scenarios"][variant] = {}

        # tqdm 진행률 표시 설정
        scenario_iter = scenarios
        if TQDM_AVAILABLE:
            scenario_iter = tqdm(
                scenarios,
                desc=f"[{variant}]",
                unit="scenario",
                leave=True,
                ncols=100,
            )

        if scenario_parallel and len(scenarios) > 1:
            # 시나리오 레벨 병렬 실행 (rate limit 대응: 워커 수 제한)
            effective_workers = scenario_max_workers
            semaphore = threading.Semaphore(effective_workers)
            completed_count = 0
            completed_lock = threading.Lock()
            pbar = scenario_iter if TQDM_AVAILABLE else None

            def run_with_logging(scenario_path, scenario_output_dir, idx):
                nonlocal completed_count
                scenario_id = scenario_path.stem
                start_time = time.time()

                # Rate limit 대응: 시작 전 딜레이
                time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

                logger.log_scenario_start(scenario_id, idx)

                result = run_single_benchmark(
                    scenario_path=scenario_path,
                    output_dir=scenario_output_dir,
                    agents=agents,
                    verbose=verbose,
                    parallel=parallel,
                    max_workers=max_workers,
                    multi_judge=multi_judge,
                )

                elapsed = time.time() - start_time
                success_count = sum(1 for r in result.get("agents", {}).values() if r.get("success"))
                logger.log_scenario_complete(scenario_id, elapsed, success_count)

                with completed_lock:
                    completed_count += 1
                    if pbar:
                        pbar.update(1)

                return scenario_id, result

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {}
                for idx, scenario_path in enumerate(scenarios):
                    scenario_output_dir = run_dir / variant / scenario_path.stem

                    future = executor.submit(
                        run_with_logging,
                        scenario_path,
                        scenario_output_dir,
                        scenario_index + idx,
                    )
                    futures[future] = scenario_path.stem
                    # Rate limit 대응: 제출 간 딜레이
                    time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

                for future in as_completed(futures):
                    scenario_id = futures[future]
                    try:
                        _, scenario_result = future.result()
                        results["scenarios"][variant][scenario_id] = scenario_result
                    except Exception as e:
                        print(f"  ❌ {scenario_id}: 예외 발생 - {e}")
                        results["scenarios"][variant][scenario_id] = {
                            "scenario": scenario_id,
                            "error": str(e),
                        }

            if pbar:
                pbar.close()
            scenario_index += len(scenarios)
        else:
            # Sequential execution with tqdm
            for idx, scenario_path in enumerate(scenario_iter):
                scenario_id = scenario_path.stem
                scenario_output_dir = run_dir / variant / scenario_id
                start_time = time.time()

                logger.log_scenario_start(scenario_id, scenario_index + idx)

                scenario_result = run_single_benchmark(
                    scenario_path=scenario_path,
                    output_dir=scenario_output_dir,
                    agents=agents,
                    verbose=verbose,
                    parallel=parallel,
                    max_workers=max_workers,
                    multi_judge=multi_judge,
                )

                elapsed = time.time() - start_time
                success_count = sum(1 for r in scenario_result.get("agents", {}).values() if r.get("success"))
                logger.log_scenario_complete(scenario_id, elapsed, success_count)

                results["scenarios"][variant][scenario_id] = scenario_result

                # Rate limit 대응: 시나리오 간 딜레이
                time.sleep(float(os.getenv("BENCHMARK_DELAY", "2.0")))

            scenario_index += len(scenarios)

    # 전체 결과 저장
    summary_path = run_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # 최종 요약 로그
    logger.log_final_summary(results)

    return results


def generate_summary_report(results: dict, output_path: Path) -> None:
    """전체 결과 요약 리포트 생성"""
    lines = [
        "# ISD Agent Benchmark 결과 요약",
        "",
        f"실행 시간: {results['timestamp']}",
        "",
        "## 설정",
        f"- Variant: {', '.join(results['config']['variants'])}",
        f"- Agent: {', '.join(results['config']['agents'])}",
        "",
        "## 결과 요약",
        "",
    ]

    # 집계
    total_scenarios = 0
    agent_stats = {}

    for variant, scenarios in results.get("scenarios", {}).items():
        lines.append(f"### {variant.upper()}")
        lines.append("")

        for scenario_id, scenario_result in scenarios.items():
            total_scenarios += 1
            lines.append(f"#### {scenario_id}")
            lines.append("")

            for agent_id, agent_result in scenario_result.get("agents", {}).items():
                if agent_id not in agent_stats:
                    agent_stats[agent_id] = {"success": 0, "failed": 0}

                if agent_result.get("success"):
                    agent_stats[agent_id]["success"] += 1
                    status = "성공"
                else:
                    agent_stats[agent_id]["failed"] += 1
                    status = f"실패: {agent_result.get('error', 'Unknown')[:50]}"

                lines.append(f"- {agent_id}: {status}")

            # 평가 결과
            if "evaluation" in scenario_result:
                eval_data = scenario_result["evaluation"]
                lines.append("")
                lines.append("**평가 점수:**")

                # Comparison rankings에서 점수 추출 (ranking 순서대로 표시)
                comparison = eval_data.get("comparison", {})
                rankings = comparison.get("rankings", [])
                
                # 순위순 정렬 보장
                rankings.sort(key=lambda x: x.get("rank", 999))
                
                for rank_info in rankings:
                    agent_id = rank_info.get("agent_id", "unknown")
                    total = rank_info.get("total_score", 0)
                    process = rank_info.get("process_score")
                    
                    score_str = f"{total:.1f}/100"
                    if process is not None:
                         score_str += f" (과정: {process:.1f})"
                         
                    lines.append(f"- {agent_id}: {score_str}")

                # 에러 등으로 rankings가 없는 경우 대비
                if not rankings and "agents" in eval_data:
                     # 기존 로직 (fallback)
                     for agent_score in eval_data.get("agents", []):
                        agent_id = agent_score.get("agent_id", "unknown")
                        total = agent_score.get("total", 0)
                        lines.append(f"- {agent_id}: {total:.1f}/100")

            lines.append("")

    # Agent 통계
    lines.append("## Agent 통계")
    lines.append("")
    lines.append("| Agent | 성공 | 실패 | 성공률 |")
    lines.append("|-------|------|------|--------|")

    for agent_id, stats in agent_stats.items():
        total = stats["success"] + stats["failed"]
        rate = (stats["success"] / total * 100) if total > 0 else 0
        lines.append(f"| {agent_id} | {stats['success']} | {stats['failed']} | {rate:.1f}% |")

    lines.append("")
    lines.append("---")
    lines.append(f"총 시나리오: {total_scenarios}개")

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """메인 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ISD Agent Benchmark 실행"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Agent 패키지 설치",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Agent 설치 상태 확인",
    )
    parser.add_argument(
        "--variant",
        "-t",
        type=str,
        default=None,
        help="실행할 variant (쉼표 구분, 예: idld_aligned,context_variant). --dataset과 함께 사용 불가",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=["train", "test"],
        default=None,
        help="데이터셋 선택 (train: 학습용, test: 평가용). --variant와 함께 사용 불가",
    )
    parser.add_argument(
        "--agents",
        "-a",
        type=str,
        default=None,
        help="실행할 Agent (쉼표 구분)",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default=None,
        help="특정 시나리오 파일 경로",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 출력",
    )
    # 병렬 실행 옵션 (기본값: 최대 병렬화)
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Agent 병렬 실행 비활성화 (기본: 병렬 실행)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=6,
        help="Agent 동시 실행 수 (기본값: 6, 모든 에이전트 동시)",
    )
    parser.add_argument(
        "--no-scenario-parallel",
        action="store_true",
        help="시나리오 레벨 병렬 실행 비활성화 (기본: 병렬 실행)",
    )
    parser.add_argument(
        "--scenario-max-workers",
        type=int,
        default=8,
        help="Concurrent scenario execution count (default: 8)",
    )
    parser.add_argument(
        "--multi-judge",
        action="store_true",
        default=True,
        help="Use multi-judge evaluation with 5 LLMs (default: True)",
    )
    parser.add_argument(
        "--single-judge",
        action="store_true",
        help="Use single-judge evaluation (disable multi-judge)",
    )
    parser.add_argument(
        "--rate-limit",
        "-r",
        type=str,
        choices=["conservative", "moderate", "aggressive", "turbo"],
        default="conservative",
        help="Rate limit mode: conservative (2x2=4), moderate (3x4=12), aggressive (6x8=48), turbo (6x16=96). Default: conservative",
    )

    args = parser.parse_args()

    # Rate limit 모드에 따른 설정 조정
    rate_limit_configs = {
        "conservative": {"max_workers": 2, "scenario_max_workers": 2, "delay": 2.0},
        "moderate": {"max_workers": 3, "scenario_max_workers": 4, "delay": 0.5},
        "aggressive": {"max_workers": 6, "scenario_max_workers": 8, "delay": 0.1},
        "turbo": {"max_workers": 6, "scenario_max_workers": 16, "delay": 0.0},
    }
    rate_config = rate_limit_configs[args.rate_limit]

    # 명시적으로 설정하지 않았으면 rate_limit 모드 값 사용
    if args.max_workers == 6:  # default value
        args.max_workers = rate_config["max_workers"]
    if args.scenario_max_workers == 8:  # default value
        args.scenario_max_workers = rate_config["scenario_max_workers"]

    # 전역 딜레이 설정
    os.environ["BENCHMARK_DELAY"] = str(rate_config["delay"])

    # 설치
    if args.install:
        success = install_agents()
        sys.exit(0 if success else 1)

    # 설치 확인
    if args.check:
        print("\nAgent 설치 상태:")
        print("-" * 40)
        status = check_agents_installed()
        for agent, installed in status.items():
            icon = "✓" if installed else "✗"
            print(f"  {icon} {agent}")

        all_installed = all(status.values())
        if not all_installed:
            print("\n일부 Agent가 설치되지 않았습니다.")
            print("--install 옵션으로 설치하세요.")
        sys.exit(0 if all_installed else 1)

    # 모듈 import 가능 여부 확인
    status = check_agents_installed()
    if not all(status.values()):
        missing = [agent for agent, installed in status.items() if not installed]
        print(f"\n⚠️  Agent 모듈 import 실패: {', '.join(missing)}")
        print("sys.path 또는 의존성을 확인하세요.")
        sys.exit(1)

    # 단일 시나리오 실행
    if args.scenario:
        scenario_path = Path(args.scenario)
        if not scenario_path.exists():
            print(f"오류: 시나리오 파일을 찾을 수 없습니다: {scenario_path}")
            sys.exit(1)

        output_dir = RESULTS_DIR / f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        agents = args.agents.split(",") if args.agents else None

        # Determine multi-judge setting
        use_multi_judge = args.multi_judge and not args.single_judge

        result = run_single_benchmark(
            scenario_path=scenario_path,
            output_dir=output_dir,
            agents=agents,
            verbose=args.verbose,
            parallel=not args.no_parallel,
            max_workers=args.max_workers,
            multi_judge=use_multi_judge,
        )

        print(f"\nResults saved: {output_dir}")
        sys.exit(0)

    # --dataset과 --variant 동시 사용 방지
    if args.dataset and args.variant:
        print("오류: --dataset과 --variant는 동시에 사용할 수 없습니다.")
        print("  --dataset: train/test 데이터셋 모드 (평가용)")
        print("  --variant: 기존 variant 모드 (idld_aligned, context_variant)")
        sys.exit(1)

    # 전체 벤치마크 실행
    variants = args.variant.split(",") if args.variant else None
    agents = args.agents.split(",") if args.agents else None

    # Determine multi-judge setting
    use_multi_judge = args.multi_judge and not args.single_judge

    results = run_full_benchmark(
        variants=variants,
        agents=agents,
        verbose=args.verbose,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
        scenario_parallel=not args.no_scenario_parallel,
        scenario_max_workers=args.scenario_max_workers,
        dataset=args.dataset,
        multi_judge=use_multi_judge,
    )

    # 요약 리포트 생성
    timestamp = results["timestamp"]
    output_dir = results.get("output_dir", RESULTS_DIR / f"benchmark_{timestamp}")
    summary_path = Path(output_dir) / "SUMMARY.md"
    generate_summary_report(results, summary_path)
    print(f"요약 리포트: {summary_path}")


if __name__ == "__main__":
    main()
