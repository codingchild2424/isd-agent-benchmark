# RPISD Agent

RPISD(Rapid Prototyping Instructional System Design) 모형 기반 래피드 프로토타이핑 교수설계 에이전트

## 개요

RPISD Agent는 임철일, 연은경(2006)의 래피드 프로토타이핑 방법론을 LangGraph StateGraph로 구현한 교수설계 에이전트입니다.

### 핵심 특징

- **이중 루프 구조**: 프로토타입 루프(설계↔사용성평가) + 개발 루프(개발↔사용성평가)
- **프로토타입 버전 관리**: 반복 개선 이력 추적
- **다중 피드백 통합**: 의뢰인/전문가/학습자 3종 피드백

## 아키텍처

```
[START] → [착수] → [분석] → [설계] ←─────── [사용성평가] (내부 루프)
                              │                    ↑
                              │                    │
                              ▼                    │
                           [개발] ─────────────────┘ (외부 루프)
                              │
                              ▼
                           [실행] → [END]
```

## 설치

```bash
cd agents/rpisd-agent
pip install -e ".[dev]"
```

## 사용법

### CLI

```bash
# 실행
rpisd-agent run scenario.json --output ./output --threshold 0.8

# 검증
rpisd-agent validate output/EASY-001_rpisd.json

# 정보
rpisd-agent info
```

### Python

```python
from rpisd_agent import RPISDAgent

agent = RPISDAgent(
    model="solar-mini",
    max_iterations=3,
    quality_threshold=0.8,
    debug=True,
)

result = agent.run(scenario)
```

## 도구 (14개)

| 단계 | 도구 | 설명 |
|------|------|------|
| 착수 | `kickoff_meeting` | 프로젝트 범위, 역할 정의 |
| 분석 | `analyze_gap` | 차이 분석 |
| 분석 | `analyze_performance` | 수행 분석 |
| 분석 | `analyze_learner_characteristics` | 학습자 특성 분석 |
| 분석 | `analyze_initial_task` | 초기 과제 분석 |
| 설계 | `design_instruction` | 교수설계 |
| 설계 | `develop_prototype` | 프로토타입 개발 (반복) |
| 설계 | `analyze_task_detailed` | 상세 과제 분석 |
| 평가 | `evaluate_with_client` | 의뢰인 평가 |
| 평가 | `evaluate_with_expert` | 전문가 평가 |
| 평가 | `evaluate_with_learner` | 학습자 평가 |
| 평가 | `aggregate_feedback` | 피드백 통합 |
| 개발 | `develop_final_program` | 최종 개발 |
| 실행 | `implement_program` | 실행 계획 |

## 출력 스키마

ADDIE 호환 출력 (`addie_output`) + RPISD 고유 출력 (`rpisd_output`)

```json
{
  "scenario_id": "EASY-001",
  "agent_id": "rpisd-agent",
  "addie_output": { ... },
  "rpisd_output": {
    "kickoff": { ... },
    "analysis": { ... },
    "design": { ... },
    "prototype_versions": [ ... ],
    "usability_feedback": { ... },
    "development": { ... },
    "implementation": { ... }
  },
  "trajectory": { ... },
  "metadata": { ... }
}
```

## 참고 자료

- 임철일, 연은경 (2006). 기업교육 프로그램 개발을 위한 사용자 중심의 래피드 프로토타입 방법론에 관한 연구. *기업교육연구*, 8(2), 27-50.
