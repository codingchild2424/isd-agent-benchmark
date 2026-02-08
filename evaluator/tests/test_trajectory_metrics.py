import sys
import os
import importlib.util

# --- MOCKING DEPENDENCIES START ---
# Mock pydantic since it is not available in the environment
from types import ModuleType

mock_pydantic = ModuleType("pydantic")
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def dict(self):
        return self.__dict__

def MockField(default=None, **kwargs):
    return default

mock_pydantic.BaseModel = MockBaseModel
mock_pydantic.Field = MockField
sys.modules["pydantic"] = mock_pydantic
# --- MOCKING DEPENDENCIES END ---

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/isd_evaluator/metrics/trajectory.py"))
trajectory_module = load_module_from_file("trajectory_evaluator", metrics_path)

TrajectoryEvaluator = trajectory_module.TrajectoryEvaluator

def test_tool_correctness():
    print("Testing Tool Correctness...", end=" ")
    evaluator = TrajectoryEvaluator()

    trajectory = {
        "tool_calls": [{"tool": "analyze_learners"}, {"tool": "design_objectives"}],
        "reasoning_steps": [
            "First, I will analyze the learners to understand their needs.",
            "Therefore, based on the analysis, I will design the learning objectives.",
            "Next, I'll move to the development phase."
        ],
        "agent_interactions": []
    }

    score = evaluator.evaluate(trajectory)

    # Check if logic works (score > 15 means it captured tool correctness)
    if score.tool_correctness > 15.0:
        print(f"PASS (Tool Correctness: {score.tool_correctness})")
    else:
        print(f"FAIL (Tool Correctness: {score.tool_correctness})")

def test_result_utilization():
    print("Testing Result Utilization...", end=" ")
    evaluator = TrajectoryEvaluator()

    trajectory = {
        "tool_calls": [
            {"tool": "analyze_learners", "result": "Found 20 beginner learners"},
            {"tool": "design_objectives", "args": {"target": "beginner learners"}},
        ],
        "reasoning_steps": [
            "Based on learner analysis, I found 20 beginner learners.",
            "Using this information, I will design objectives for beginners."
        ],
        "agent_interactions": []
    }

    score = evaluator.evaluate(trajectory)

    # 결과를 다음 단계에 잘 활용했는지 평가
    if score.result_utilization >= 15.0:
        print(f"PASS (Result Utilization: {score.result_utilization})")
    else:
        print(f"FAIL (Result Utilization: {score.result_utilization})")

def test_redundancy_avoidance():
    print("Testing Redundancy Avoidance...", end=" ")
    evaluator = TrajectoryEvaluator()

    # 중복 호출이 있는 비효율적인 궤적
    trajectory = {
        "tool_calls": [
            {"tool": "analyze_learners"},
            {"tool": "analyze_learners"},  # 중복
            {"tool": "design_objectives"},
        ],
        "reasoning_steps": [],
        "agent_interactions": []
    }

    score = evaluator.evaluate(trajectory)

    # 중복이 있으면 점수가 낮아야 함
    print(f"INFO (Redundancy Avoidance: {score.redundancy_avoidance})")

if __name__ == "__main__":
    try:
        test_tool_correctness()
        test_result_utilization()
        test_redundancy_avoidance()
        print("\nAll tests completed.")
    except Exception as e:
        print(f"\nTest Execution Failed: {e}")
        import traceback
        traceback.print_exc()
