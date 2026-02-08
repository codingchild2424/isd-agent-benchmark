"""
pytest 설정 파일

ADDIE 루브릭 평가 테스트를 위한 fixture 정의
- LLM 응답 Mocking
- 샘플 데이터
- 공통 유틸리티
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))


# ============================================================================
# 경로 Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def fixtures_dir():
    """fixtures 디렉토리 경로"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def llm_responses_dir(fixtures_dir):
    """LLM 응답 fixture 디렉토리 경로"""
    return fixtures_dir / "llm_responses"


# ============================================================================
# Fixture 로더
# ============================================================================

@pytest.fixture
def load_fixture(fixtures_dir):
    """JSON fixture 파일 로더"""
    def _load(filename: str) -> dict:
        filepath = fixtures_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return _load


@pytest.fixture
def load_llm_response(llm_responses_dir):
    """LLM 응답 fixture 로더 (문자열 반환)

    JSON 파일인 경우 마크다운 코드 펜스로 감싸서 반환
    (실제 LLM 응답 형식과 동일하게)
    """
    def _load(filename: str) -> str:
        filepath = llm_responses_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # JSON 파일인 경우 마크다운 코드 펜스로 감싸기
        if filename.endswith('.json'):
            return f"```json\n{content}\n```"
        return content
    return _load


@pytest.fixture
def load_raw_json(llm_responses_dir):
    """LLM 응답 fixture 로더 (JSON 파싱)"""
    def _load(filename: str) -> dict:
        filepath = llm_responses_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return _load


# ============================================================================
# Mock Evaluator Fixtures
# ============================================================================

@pytest.fixture
def mock_evaluator():
    """_call_llm이 mocking된 ADDIERubricEvaluator 생성

    실제 LLM API 호출 없이 테스트 가능
    """
    from isd_evaluator.metrics.addie_rubric import ADDIERubricEvaluator

    # API 키 없이 생성 (테스트용)
    evaluator = ADDIERubricEvaluator(
        provider="upstage",
        model="solar-pro2-251215",
        api_key="test-api-key-for-unit-test",
        include_benchmarks=False,  # 벤치마크 예시 비활성화 (토큰 절약)
    )

    # _call_llm 메서드를 기본 Mock으로 설정
    evaluator._call_llm = MagicMock(return_value='{"sub_scores": {}, "sub_reasoning": {}}')

    return evaluator


@pytest.fixture
def mock_llm_response():
    """_call_llm 메서드를 특정 응답으로 mocking하는 헬퍼

    사용 예:
        evaluator = mock_llm_response(evaluator, response_string)
    """
    def _mock(evaluator, response: str):
        evaluator._call_llm = MagicMock(return_value=response)
        return evaluator
    return _mock


# ============================================================================
# 샘플 데이터 Fixtures
# ============================================================================

@pytest.fixture
def sample_scenario():
    """테스트용 샘플 시나리오"""
    return {
        "title": "파이썬 기초 프로그래밍",
        "context": {
            "target_audience": "대학생",
            "prior_knowledge": "프로그래밍 경험 없음",
            "duration": "1학기",
            "learning_environment": "온라인",
        },
        "learning_goals": ["변수와 자료형 이해", "조건문과 반복문 작성"],
        "constraints": {},
    }


@pytest.fixture
def sample_addie_output():
    """테스트용 ADDIE 산출물"""
    return {
        "analysis": {
            "learner_analysis": "초보자 대상, 프로그래밍 경험 없음",
            "context_analysis": "온라인 학습 환경",
            "needs_analysis": "기초 프로그래밍 역량 필요",
        },
        "design": {
            "learning_objectives": "ABCD 모델 기반 학습목표",
            "assessment_design": "형성평가 및 총괄평가 계획",
            "instructional_strategies": "단계적 학습, 실습 중심",
        },
        "development": {
            "content": "학습 자료 개발 계획",
            "materials": "교수자용/학습자용 자료",
        },
        "implementation": {
            "delivery_plan": "온라인 전달 계획",
            "instructor_guide": "교수자 가이드",
        },
        "evaluation": {
            "assessment": "평가 도구",
            "improvement_plan": "개선 계획",
        },
    }


# ============================================================================
# 단계별 Sub-item ID Fixtures
# ============================================================================

@pytest.fixture
def analysis_sub_items():
    """Analysis 단계 소단계 ID 목록"""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def design_sub_items():
    """Design 단계 소단계 ID 목록"""
    return [11, 12, 13, 14, 15, 16, 17, 18]


@pytest.fixture
def development_sub_items():
    """Development 단계 소단계 ID 목록"""
    return [19, 20, 21, 22, 23]


@pytest.fixture
def implementation_sub_items():
    """Implementation 단계 소단계 ID 목록"""
    return [24, 25, 26, 27]


@pytest.fixture
def evaluation_sub_items():
    """Evaluation 단계 소단계 ID 목록"""
    return [28, 29, 30, 31, 32, 33]


@pytest.fixture
def all_sub_items():
    """33개 전체 소단계 ID 목록"""
    return list(range(1, 34))


# ============================================================================
# ITEM_MAPPING Fixtures (테스트 검증용)
# ============================================================================

@pytest.fixture
def item_mapping():
    """13개 항목 -> 33개 소단계 매핑 (CSV 중단계 기준)"""
    return {
        # Analysis (A1, A2, A3) - 분석 단계 (3개 중단계)
        "A1": [1, 2, 3, 4],         # 요구분석
        "A2": [5, 6],               # 학습자 및 환경분석
        "A3": [7, 8, 9, 10],        # 과제 및 목표분석
        # Design (D1, D2, D3) - 설계 단계 (3개 중단계)
        "D1": [11, 12],             # 평가 및 목표 정렬 설계
        "D2": [13, 14, 15, 16, 17], # 교수전략 및 학습경험 설계
        "D3": [18],                 # 프로토타입 구조 설계
        # Development (Dev1, Dev2) - 개발 단계 (2개 중단계)
        "Dev1": [19, 20, 21, 22],   # 프로토타입 개발
        "Dev2": [23],               # 개발 결과 검토 및 수정
        # Implementation (I1, I2) - 실행 단계 (2개 중단계)
        "I1": [24, 25],             # 프로그램 실행 준비
        "I2": [26, 27],             # 프로그램 실행
        # Evaluation (E1, E2, E3) - 평가 단계 (3개 중단계)
        "E1": [28, 29],             # 형성평가
        "E2": [30, 31, 32],         # 총괄평가 및 채택 결정
        "E3": [33],                 # 프로그램 개선 및 환류
    }


# ============================================================================
# 완전한 sub_scores Fixtures
# ============================================================================

@pytest.fixture
def complete_sub_scores():
    """33개 전체 sub_score (모두 7.5점)"""
    return {i: 7.5 for i in range(1, 34)}


@pytest.fixture
def complete_sub_reasoning():
    """33개 전체 reasoning"""
    return {i: f"항목 {i} 평가 완료" for i in range(1, 34)}


@pytest.fixture
def varying_sub_scores():
    """다양한 점수 분포 (단계별로 다름)"""
    scores = {}
    # Analysis: 높은 점수 (9.0)
    for i in range(1, 11):
        scores[i] = 9.0
    # Design: 중간 점수 (7.0)
    for i in range(11, 19):
        scores[i] = 7.0
    # Development: 낮은 점수 (5.0)
    for i in range(19, 24):
        scores[i] = 5.0
    # Implementation: 중간 점수 (6.0)
    for i in range(24, 28):
        scores[i] = 6.0
    # Evaluation: 높은 점수 (8.5)
    for i in range(28, 34):
        scores[i] = 8.5
    return scores
