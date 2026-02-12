# EduPlanner Agent

EduPlanner 논문 기반 3-Agent 협업 교수설계 시스템

## 개요

EduPlanner는 3개의 전문 Agent가 적대적 협업(Adversarial Collaboration)을 통해
고품질의 ADDIE 기반 교수설계 산출물을 생성합니다.

## Agent 구조

1. **Generator Agent**: 초기 ADDIE 산출물 생성
2. **Evaluator Agent**: ADDIE Rubric 13항목 평가
3. **Optimizer Agent**: 피드백 기반 최적화
4. **Analyst Agent**: 오류 분석 및 개선점 도출

## 설치

```bash
pip install -e .
```

## 사용법

```bash
# 기본 실행
eduplanner run --input scenario.json --output result.json

# 궤적 저장
eduplanner run --input scenario.json --output result.json --trajectory traj.json

# 정보 출력
eduplanner info
```

## CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--input, -i` | 입력 시나리오 JSON 파일 |
| `--output, -o` | 출력 ADDIE 산출물 파일 |
| `--trajectory, -t` | 궤적 저장 파일 (선택) |
| `--max-iterations` | 최대 반복 횟수 (기본: 2) |
| `--verbose, -v` | 상세 출력 |

## 참고

- 논문: EduPlanner (Zhang et al., 2025)
- ADDIE Rubric 평가 기준: Analysis(3), Design(3), Development(3), Implementation(2), Evaluation(2) - 총 13항목

## 구현 검증 (Verification & Alignment)

본 구현체는 EduPlanner 논문의 아키텍처를 계승하되, **ISD Agent Benchmark**의 목적에 맞춰 적용 범위를 확장하였습니다.

### 원본과의 공통점 (Aligned)
- **3-Agent 협업 구조**: Evaluator, Optimizer, Analyst(Question Analyst 계승)의 3자 협업 체제 유지
- **Skill-Tree 모델링**: 학습자의 역량을 다차원적으로 모델링하는 구조 채택
- **프로세스**: 생성 -> 평가 -> 최적화의 반복 루프 구조

### 시나리오 리더보드 맞춤형 변형 (Adapted)
- **적용 도메인 확장**: 수학 수업 지도안(Math Lesson Plan) -> 포괄적 ADDIE 교수설계(Instructional Systems Design)
- **Skill-Tree 항목**: 수학적 능력(연산, 추론 등) -> 교수설계 맥락(사전지식, 동기, 기술활용성 등)으로 재정의
- **평가 기준**: 자체 기준 -> **ADDIE Rubric** (13개 세부 항목) 프레임워크 적용

