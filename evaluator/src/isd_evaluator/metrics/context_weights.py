"""
컨텍스트 인식형 평가 가중치 조정

시나리오의 맥락(학습자 특성, 기관 유형, 전달 방식, 제약 조건)에 따라
ADDIE 단계별 가중치를 동적으로 조정합니다.
"""

from typing import Dict, Optional
from isd_evaluator.models import ADDIEPhase, ContextProfile
from isd_evaluator.rubrics.addie_definitions import DEFAULT_PHASE_WEIGHTS


class ContextWeightAdjuster:
    """컨텍스트 기반 가중치 조정기"""

    # 기본 가중치
    BASE_WEIGHTS = DEFAULT_PHASE_WEIGHTS.copy()

    # 컨텍스트별 가중치 조정 규칙
    ADJUSTMENT_RULES = {
        # 기관 유형별 조정
        "institution_type": {
            "기업": {
                ADDIEPhase.ANALYSIS: 0.05,       # 요구분석 중시
                ADDIEPhase.IMPLEMENTATION: 0.05, # 실행 가능성 중시
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "대학교(학부)": {
                ADDIEPhase.DESIGN: 0.05,         # 교수전략 중시
                ADDIEPhase.EVALUATION: 0.05,     # 평가 체계 중시
                ADDIEPhase.IMPLEMENTATION: -0.05,
                ADDIEPhase.ANALYSIS: -0.05,
            },
            "대학": {
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.EVALUATION: 0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
                ADDIEPhase.ANALYSIS: -0.05,
            },
            "초·중등학교": {
                ADDIEPhase.DEVELOPMENT: 0.10,    # 학습자료 품질 중시
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "초등학교": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "중학교": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "고등학교": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
        },
        # 전달 방식별 조정
        "delivery_mode": {
            "온라인 비실시간(LMS)": {
                ADDIEPhase.DEVELOPMENT: 0.10,    # 콘텐츠 품질 중시
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "온라인 비실시간": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "LMS": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "블렌디드(혼합형)": {
                ADDIEPhase.DESIGN: 0.05,         # 설계 복잡도 반영
                ADDIEPhase.IMPLEMENTATION: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "블렌디드": {
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.IMPLEMENTATION: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "혼합형": {
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.IMPLEMENTATION: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "시뮬레이션/VR 기반": {
                ADDIEPhase.DEVELOPMENT: 0.15,    # 기술적 개발 중시
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "VR": {
                ADDIEPhase.DEVELOPMENT: 0.15,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "시뮬레이션": {
                ADDIEPhase.DEVELOPMENT: 0.15,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            # 오프라인(교실 수업) - Context Matrix 33번
            "오프라인(교실 수업)": {
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 학습자 상호작용, 현장 진행 중시
                ADDIEPhase.DESIGN: 0.05,          # 활동 설계 중시
                ADDIEPhase.DEVELOPMENT: -0.10,    # 디지털 콘텐츠 의존도 낮음
                ADDIEPhase.ANALYSIS: -0.05,
            },
            "오프라인": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.05,
            },
            "대면": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.05,
            },
            "교실": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.05,
            },
            # 온라인 실시간(Zoom 등) - Context Matrix 34번
            "온라인 실시간(Zoom 등)": {
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 실시간 진행 역량 중시
                ADDIEPhase.DESIGN: 0.05,          # 참여 유도 설계 필요
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "온라인 실시간": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "실시간": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "Zoom": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            # 모바일 마이크로러닝 - Context Matrix 37번
            "모바일 마이크로러닝": {
                ADDIEPhase.DEVELOPMENT: 0.15,     # 짧고 집중적인 콘텐츠 품질 중시
                ADDIEPhase.ANALYSIS: 0.05,        # 모바일 학습자 특성 분석 중요
                ADDIEPhase.IMPLEMENTATION: -0.10,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "모바일": {
                ADDIEPhase.DEVELOPMENT: 0.15,
                ADDIEPhase.ANALYSIS: 0.05,
                ADDIEPhase.IMPLEMENTATION: -0.10,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "마이크로러닝": {
                ADDIEPhase.DEVELOPMENT: 0.15,
                ADDIEPhase.ANALYSIS: 0.05,
                ADDIEPhase.IMPLEMENTATION: -0.10,
                ADDIEPhase.EVALUATION: -0.10,
            },
            # 프로젝트 기반(PBL) - Context Matrix 39번
            "프로젝트 기반(PBL)": {
                ADDIEPhase.DESIGN: 0.10,          # 프로젝트 구조화 설계 중시
                ADDIEPhase.EVALUATION: 0.10,      # 과정 기반 평가 중시
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.10,
            },
            "PBL": {
                ADDIEPhase.DESIGN: 0.10,
                ADDIEPhase.EVALUATION: 0.10,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.10,
            },
            "프로젝트": {
                ADDIEPhase.DESIGN: 0.10,
                ADDIEPhase.EVALUATION: 0.10,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.10,
            },
        },
        # 평가 요구별 조정
        "evaluation_focus": {
            "형성평가 중심": {
                ADDIEPhase.EVALUATION: 0.10,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "형성평가": {
                ADDIEPhase.EVALUATION: 0.10,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "총괄평가 중심": {
                ADDIEPhase.EVALUATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,         # 평가 설계 중시
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.DEVELOPMENT: -0.05,
            },
            "총괄평가": {
                ADDIEPhase.EVALUATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.DEVELOPMENT: -0.05,
            },
        },
        # 시간 제약별 조정
        "duration": {
            "단기 집중 과정(1주 내)": {
                ADDIEPhase.IMPLEMENTATION: 0.10, # 실행 효율성 중시
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "단기": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "1주": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "장기 과정(1~6개월)": {
                ADDIEPhase.ANALYSIS: 0.05,       # 철저한 분석 필요
                ADDIEPhase.EVALUATION: 0.05,     # 지속적 평가 필요
                ADDIEPhase.IMPLEMENTATION: -0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
            },
            "장기": {
                ADDIEPhase.ANALYSIS: 0.05,
                ADDIEPhase.EVALUATION: 0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
            },
        },
        # 학습자 연령별 조정 (Context Matrix 1-4번)
        "learner_age": {
            "10대": {
                ADDIEPhase.DEVELOPMENT: 0.10,    # 흥미로운 학습자료 중시
                ADDIEPhase.DESIGN: 0.05,         # 참여형 활동 설계 중시
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "20대": {
                # 기본 가중치 유지 (대학생 기준 설계됨)
            },
            "30대": {
                ADDIEPhase.ANALYSIS: 0.05,       # 실무 요구 분석 중시
                ADDIEPhase.IMPLEMENTATION: 0.05, # 현장 적용성 중시
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "40대 이상": {
                ADDIEPhase.ANALYSIS: 0.10,       # 업무 연계 분석 중시
                ADDIEPhase.IMPLEMENTATION: 0.05, # 실무 전이 중시
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.DESIGN: -0.05,
            },
        },
        # 학력수준별 조정 (Context Matrix 5-9번)
        "learner_education": {
            "초등": {
                ADDIEPhase.DEVELOPMENT: 0.15,    # 시각자료, 게임요소 중시
                ADDIEPhase.DESIGN: 0.05,         # 활동 중심 설계
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "중등": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "고등": {
                ADDIEPhase.DESIGN: 0.05,         # 학습전략 중시
                ADDIEPhase.EVALUATION: 0.05,     # 평가 체계 중시
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "대학": {
                # 기본 가중치 유지 (기준점)
            },
            "성인": {
                ADDIEPhase.ANALYSIS: 0.10,       # 요구 분석 심화
                ADDIEPhase.IMPLEMENTATION: 0.05, # 현장 적용성 중시
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.DESIGN: -0.05,
            },
        },
        # 도메인 전문성 수준별 조정 (Context Matrix 10-12번)
        "domain_expertise": {
            "초급": {
                ADDIEPhase.DEVELOPMENT: 0.10,    # 기초 자료 품질 중시
                ADDIEPhase.DESIGN: 0.05,         # 단계적 설계 중시
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            "중급": {
                # 기본 가중치 유지
            },
            "고급": {
                ADDIEPhase.ANALYSIS: 0.10,       # 심화 요구분석 중시
                ADDIEPhase.EVALUATION: 0.10,     # 고급 평가 중시
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.DESIGN: -0.10,
            },
        },
        # 교육 도메인별 조정 (Context Matrix 23-32번)
        "education_domain": {
            # 23: 언어 - 말하기/듣기 실습 중시
            "언어": {
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 말하기/듣기 연습 중시
                ADDIEPhase.DESIGN: 0.05,          # 활동 설계 중시
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            # 24: 수학 - 개념 구조화, 단계적 설계 중시
            "수학": {
                ADDIEPhase.DESIGN: 0.10,          # 개념 구조화 설계 중시
                ADDIEPhase.DEVELOPMENT: 0.05,     # 시각화 자료 중시
                ADDIEPhase.IMPLEMENTATION: -0.10,
                ADDIEPhase.ANALYSIS: -0.05,
            },
            # 25: 과학 - 실험 설계, 자료 개발 중시
            "과학": {
                ADDIEPhase.DEVELOPMENT: 0.10,     # 실험/시각화 자료 중시
                ADDIEPhase.DESIGN: 0.05,          # 탐구 활동 설계
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.EVALUATION: -0.10,
            },
            # 26: 사회 - 분석적 사고, 토론 설계 중시
            "사회": {
                ADDIEPhase.ANALYSIS: 0.05,        # 분석적 사고 중시
                ADDIEPhase.DESIGN: 0.05,          # 토론/협력 활동 설계
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            # 27: 개발(Software/IT) - 실습 환경 및 실행 중시
            "개발(Software/IT)": {
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 실습 환경 실행 중시
                ADDIEPhase.DEVELOPMENT: 0.05,     # 실습 자료 개발
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            # 28: AI - 실습 및 개발 중시 (개발과 유사)
            "AI": {
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 실습 환경 실행 중시
                ADDIEPhase.DEVELOPMENT: 0.05,     # 실습 자료 개발
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            # 29: 의료/간호 - 절차 정확성, 평가 엄격
            "의료/간호": {
                ADDIEPhase.EVALUATION: 0.10,      # 절차 정확성 평가 중시
                ADDIEPhase.DEVELOPMENT: 0.05,     # 정확한 자료 개발
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.DESIGN: -0.10,
            },
            # 30: 경영/HR/경영지원 - 요구분석, 성과 평가 중시
            "경영/HR/경영지원": {
                ADDIEPhase.ANALYSIS: 0.05,        # 조직 요구분석 중시
                ADDIEPhase.EVALUATION: 0.05,      # 성과 평가 중시
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            # 31: 교육(교수·학습) - 교수법 설계, 평가 체계 중시
            "교육(교수·학습)": {
                ADDIEPhase.DESIGN: 0.10,          # 교수법 설계 중시
                ADDIEPhase.EVALUATION: 0.05,      # 학습 평가 체계
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.ANALYSIS: -0.10,
            },
            # 32: 서비스/고객응대 - 롤플레이, 피드백 중시
            "서비스/고객응대": {
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 롤플레이 실습 중시
                ADDIEPhase.EVALUATION: 0.05,      # 피드백 평가 중시
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.DEVELOPMENT: -0.05,
            },
        },
        # 학습자 직업/역할별 조정 (Context Matrix 13-16번)
        "learner_role": {
            # 13: 학생 - 동기부여 전략 + 학습자료 의존도 높음
            "학생": {
                ADDIEPhase.DESIGN: 0.05,          # 동기부여/활동 설계 중시
                ADDIEPhase.DEVELOPMENT: 0.05,     # 학습자료 의존도 높음
                ADDIEPhase.ANALYSIS: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            # 14: 직장인(사무/관리) - 현업 적용성 최우선
            "직장인(사무/관리)": {
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 현업 전이/적용성 중시
                ADDIEPhase.ANALYSIS: 0.05,        # 업무 요구분석 중시
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "직장인": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.ANALYSIS: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "사무직": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.ANALYSIS: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            # 15: 전문직(의료/법률/기술) - 사례 중심 + 정확성 평가
            "전문직(의료/법률/기술)": {
                ADDIEPhase.DESIGN: 0.05,          # 사례/문제 중심 설계
                ADDIEPhase.EVALUATION: 0.05,      # 전문성/정확성 평가 중시
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "전문직": {
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.EVALUATION: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.05,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            # 16: 예비 교사/교사 - 교수설계 역량 모델링 필수
            "예비 교사/교사": {
                ADDIEPhase.DESIGN: 0.10,          # 교수설계 역량 모델링
                ADDIEPhase.EVALUATION: 0.05,      # 평가 체계 이해
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.05,
            },
            "교사": {
                ADDIEPhase.DESIGN: 0.10,
                ADDIEPhase.EVALUATION: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.05,
            },
            "예비교사": {
                ADDIEPhase.DESIGN: 0.10,
                ADDIEPhase.EVALUATION: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.ANALYSIS: -0.05,
            },
        },
        # 학습자 규모별 조정 (Context Matrix 40-42번)
        "class_size": {
            # 40: 소규모(1-10명) - 개별화 학습, 개별 피드백 중시
            "소규모(1-10명)": {
                ADDIEPhase.DESIGN: 0.05,          # 개별화 설계 중시
                ADDIEPhase.EVALUATION: 0.10,      # 개별 피드백 평가 중시
                ADDIEPhase.DEVELOPMENT: -0.10,    # 표준화 자료 필요성 낮음
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "소규모": {
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.EVALUATION: 0.10,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            # 41: 중규모(10-30명) - 소집단 활동 중시
            "중규모(10-30명)": {
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 소집단 활동 운영 중시
                ADDIEPhase.DESIGN: 0.05,          # 협력 활동 설계
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "중규모": {
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            # 42: 대규모(30명 이상) - 표준화 자료, 효율성 중시
            "대규모(30명 이상)": {
                ADDIEPhase.DEVELOPMENT: 0.10,     # 표준화 자료 개발 중시
                ADDIEPhase.IMPLEMENTATION: 0.05,  # 효율적 운영 중시
                ADDIEPhase.EVALUATION: -0.10,     # 개별 피드백 어려움
                ADDIEPhase.DESIGN: -0.05,
            },
            "대규모": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.IMPLEMENTATION: 0.05,
                ADDIEPhase.EVALUATION: -0.10,
                ADDIEPhase.DESIGN: -0.05,
            },
        },
        # 기술 환경별 조정 (Context Matrix 46-48번)
        "tech_environment": {
            # 46: 디지털 기기 제공 - 멀티미디어 활용 극대화
            "디지털 기기 제공": {
                ADDIEPhase.DEVELOPMENT: 0.10,     # 멀티미디어 콘텐츠 개발 중시
                ADDIEPhase.DESIGN: 0.05,          # 기술 활용 설계
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            "디지털": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.DESIGN: 0.05,
                ADDIEPhase.ANALYSIS: -0.10,
                ADDIEPhase.EVALUATION: -0.05,
            },
            # 47: 개인 기기 지참(BYOD) - 접근성 설계 중시
            "개인 기기 지참(BYOD)": {
                ADDIEPhase.DESIGN: 0.10,          # 다양한 기기 접근성 설계
                ADDIEPhase.ANALYSIS: 0.05,        # 기기 환경 분석 필요
                ADDIEPhase.DEVELOPMENT: -0.10,    # 표준화 어려움
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            "BYOD": {
                ADDIEPhase.DESIGN: 0.10,
                ADDIEPhase.ANALYSIS: 0.05,
                ADDIEPhase.DEVELOPMENT: -0.10,
                ADDIEPhase.IMPLEMENTATION: -0.05,
            },
            # 48: 제한적 기술 환경 - 저기술 대안, 오프라인 자료 중시
            "제한적 기술 환경": {
                ADDIEPhase.DEVELOPMENT: 0.10,     # 저기술 대안 자료 개발
                ADDIEPhase.IMPLEMENTATION: 0.10,  # 오프라인 운영 역량
                ADDIEPhase.DESIGN: -0.10,         # 기술 의존 설계 불가
                ADDIEPhase.ANALYSIS: -0.10,
            },
            "제한적": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: -0.10,
                ADDIEPhase.ANALYSIS: -0.10,
            },
            "저기술": {
                ADDIEPhase.DEVELOPMENT: 0.10,
                ADDIEPhase.IMPLEMENTATION: 0.10,
                ADDIEPhase.DESIGN: -0.10,
                ADDIEPhase.ANALYSIS: -0.10,
            },
        },
    }

    def __init__(self, context_profile: Optional[ContextProfile] = None):
        self.context_profile = context_profile

    def get_adjusted_weights(self) -> Dict[ADDIEPhase, float]:
        """컨텍스트에 따라 조정된 가중치 반환"""
        weights = self.BASE_WEIGHTS.copy()

        if not self.context_profile:
            return weights

        # 각 컨텍스트 요소별 조정 적용
        for attr, rules in self.ADJUSTMENT_RULES.items():
            value = getattr(self.context_profile, attr, None)
            if value:
                # 정확히 일치하는 규칙 찾기
                if value in rules:
                    adjustments = rules[value]
                    for phase, delta in adjustments.items():
                        weights[phase] = max(0.05, min(0.50, weights[phase] + delta))
                else:
                    # 부분 일치 찾기
                    for rule_key, adjustments in rules.items():
                        if rule_key.lower() in value.lower() or value.lower() in rule_key.lower():
                            for phase, delta in adjustments.items():
                                weights[phase] = max(0.05, min(0.50, weights[phase] + delta))
                            break

        # 정규화 (합이 1.0이 되도록)
        total = sum(weights.values())
        return {phase: w / total for phase, w in weights.items()}

    @classmethod
    def from_scenario(cls, scenario: dict) -> "ContextWeightAdjuster":
        """시나리오에서 컨텍스트 프로필 추출 (v2 - 신규 필드 우선 참조)"""
        context = scenario.get("context", {})
        constraints = scenario.get("constraints", {})

        # 컨텍스트 매핑 (신규 필드 우선, 폴백으로 기존 추론 로직 사용)
        profile = ContextProfile(
            # 기존 필드
            institution_type=context.get("institution_type") or cls._infer_institution_type(context),
            delivery_mode=context.get("learning_environment"),
            duration=cls._categorize_duration(context.get("duration")),
            # 평가 요구 (신규 필드 우선)
            evaluation_focus=constraints.get("assessment_type") or cls._infer_evaluation_focus(constraints),
            # 학습자 특성 필드 (신규 필드 우선, Context Matrix 1-16번)
            learner_age=context.get("learner_age") or cls._infer_learner_age(context),
            learner_education=context.get("learner_education") or cls._infer_education_level(context),
            domain_expertise=context.get("domain_expertise") or cls._infer_domain_expertise(context),  # Items 10-12
            learner_role=context.get("learner_role") or cls._infer_learner_role(context),  # Items 13-16
            # 교육 도메인 (Context Matrix 23-32번)
            education_domain=scenario.get("domain") or cls._infer_education_domain(context),
            # 제약 조건 (신규 필드 우선, Context Matrix 40-42, 46-48번)
            class_size=cls._normalize_class_size(context.get("class_size")),  # Items 40-42
            tech_environment=constraints.get("tech_requirements") or cls._infer_tech_environment(context),  # Items 46-48
        )

        return cls(profile)

    @staticmethod
    def _normalize_class_size(size) -> Optional[str]:
        """class_size를 범주형 문자열로 정규화 (정수 또는 문자열 모두 처리)"""
        if size is None:
            return None
        # 이미 문자열이면 그대로 반환
        if isinstance(size, str):
            return size
        # 정수이면 범주형으로 변환
        if isinstance(size, int):
            if size <= 10:
                return "소규모(1-10명)"
            elif size <= 30:
                return "중규모(10-30명)"
            else:
                return "대규모(30명 이상)"
        return None

    @staticmethod
    def _infer_institution_type(context: dict) -> Optional[str]:
        """대상 정보에서 기관 유형 추론"""
        target = context.get("target_audience", "").lower()

        if "기업" in target or "직장" in target or "마케팅" in target or "신입" in target:
            return "기업"
        elif "대학" in target or "학부" in target or "대학생" in target:
            return "대학교(학부)"
        elif "초등" in target:
            return "초등학교"
        elif "중학" in target or "중등" in target:
            return "중학교"
        elif "고등" in target or "고교" in target:
            return "고등학교"
        elif "학교" in target or "학생" in target:
            return "초·중등학교"

        return None

    @staticmethod
    def _categorize_duration(duration: Optional[str]) -> Optional[str]:
        """시간을 카테고리로 분류"""
        if not duration:
            return None

        duration_lower = duration.lower()

        if "1주" in duration_lower or "3일" in duration_lower or "1일" in duration_lower or "2일" in duration_lower:
            return "단기"
        elif "개월" in duration_lower or "6주" in duration_lower or "8주" in duration_lower:
            return "장기"

        # 시간 단위 파싱 시도
        import re
        hours_match = re.search(r"(\d+)\s*시간", duration_lower)
        if hours_match:
            hours = int(hours_match.group(1))
            if hours <= 40:  # 약 1주
                return "단기"

        return "중기"

    @staticmethod
    def _infer_evaluation_focus(constraints: dict) -> Optional[str]:
        """제약조건에서 평가 초점 추론"""
        if not constraints:
            return None

        constraints_str = str(constraints).lower()

        if "형성평가" in constraints_str or "형성 평가" in constraints_str:
            return "형성평가"
        elif "총괄평가" in constraints_str or "총괄 평가" in constraints_str:
            return "총괄평가"

        return None

    @staticmethod
    def _infer_learner_age(context: dict) -> Optional[str]:
        """대상 정보에서 연령대 추론 (Context Matrix 1-4번)"""
        target = context.get("target_audience", "").lower()

        if any(kw in target for kw in ["초등", "어린이", "아동", "10세", "11세", "12세"]):
            return "10대"
        elif any(kw in target for kw in ["중학", "고등", "청소년", "13세", "14세", "15세", "16세", "17세", "18세"]):
            return "10대"
        elif any(kw in target for kw in ["대학", "20대", "대학생"]):
            return "20대"
        elif any(kw in target for kw in ["신입", "30대", "주니어", "사원"]):
            return "30대"
        elif any(kw in target for kw in ["경력", "40대", "50대", "시니어", "관리자", "임원", "베테랑"]):
            return "40대 이상"

        return None

    @staticmethod
    def _infer_education_level(context: dict) -> Optional[str]:
        """대상 정보에서 학력수준 추론 (Context Matrix 5-9번)"""
        target = context.get("target_audience", "").lower()

        if "초등" in target:
            return "초등"
        elif any(kw in target for kw in ["중학", "중등"]):
            return "중등"
        elif any(kw in target for kw in ["고등", "고교"]):
            return "고등"
        elif any(kw in target for kw in ["대학", "학부", "대학생"]):
            return "대학"
        elif any(kw in target for kw in ["직장인", "성인", "신입", "경력", "직원", "사원"]):
            return "성인"

        return None

    @staticmethod
    def _infer_domain_expertise(context: dict) -> Optional[str]:
        """사전지식 정보에서 도메인 전문성 추론 (Context Matrix 10-12번)"""
        prior = context.get("prior_knowledge", "").lower()

        if any(kw in prior for kw in ["없음", "초보", "기초", "입문", "경험 없", "처음", "전무"]):
            return "초급"
        elif any(kw in prior for kw in ["기본", "중급", "어느 정도", "1년", "2년", "약간"]):
            return "중급"
        elif any(kw in prior for kw in ["고급", "전문", "숙련", "다년간", "경력", "풍부", "5년", "10년"]):
            return "고급"

        return None

    @staticmethod
    def _infer_education_domain(context: dict) -> Optional[str]:
        """시나리오에서 교육 도메인 추론 (Context Matrix 23-32번)

        CSV 항목 정확히 반영:
        23: 언어, 24: 수학, 25: 과학, 26: 사회
        27: 개발(Software/IT), 28: AI, 29: 의료/간호
        30: 경영/HR/경영지원, 31: 교육(교수·학습), 32: 서비스/고객응대
        """
        # 분석 대상 텍스트 수집
        sources = [
            context.get("topic", ""),
            context.get("subject", ""),
            context.get("title", ""),
        ]
        # objectives와 skills_to_acquire가 리스트인 경우 처리
        objectives = context.get("objectives", [])
        if isinstance(objectives, list):
            sources.extend(objectives)
        elif isinstance(objectives, str):
            sources.append(objectives)

        skills = context.get("skills_to_acquire", [])
        if isinstance(skills, list):
            sources.extend(skills)
        elif isinstance(skills, str):
            sources.append(skills)

        combined = " ".join(str(s) for s in sources).lower()

        # 도메인별 키워드 매핑 (우선순위 고려하여 순서 조정)
        # 주의: "개발"이 "리더십 개발" 등에서 오인식되지 않도록
        # 경영/HR을 개발(IT)보다 먼저 검사
        domain_keywords = {
            # 23: 언어 - "비즈니스 영어" 등 언어 학습이 주 목적일 경우 우선순위 높임
            "언어": ["영어", "회화", "작문", "문법", "한국어", "어휘", "독해", "외국어",
                   "language", "english", "speaking", "writing", "리터러시"],
            # 30: 경영/HR/경영지원 - "리더십 개발" 등 오인식 방지를 위해 우선 검사
            "경영/HR/경영지원": ["리더십", "마케팅", "경영", "인사", "hr", "management",
                             "leadership", "marketing", "조직", "전략", "재무", "회계",
                             "비즈니스", "business", "기획", "영업"],
            # 28: AI - "AI 개발" 등 정확한 분류를 위해 IT보다 우선
            "AI": ["인공지능", "ai", "머신러닝", "딥러닝", "machine learning",
                  "deep learning", "신경망", "neural", "자연어처리", "nlp", "llm"],
            # 27: 개발(Software/IT)
            "개발(Software/IT)": ["코딩", "프로그래밍", "파이썬", "소프트웨어",
                                "python", "java", "coding", "programming", "웹개발",
                                "앱개발", "software", "시스템", "데이터베이스",
                                "소프트웨어 개발", "앱 개발", "웹 개발"],
            # 24: 수학
            "수학": ["수학", "통계", "확률", "기하", "대수", "미적분", "calculus",
                   "math", "statistics", "algebra", "수치"],
            # 25: 과학
            "과학": ["물리", "화학", "생물", "과학", "실험", "science", "physics",
                   "chemistry", "biology", "지구과학", "천문"],
            # 26: 사회
            "사회": ["역사", "사회", "경제학", "정치", "지리", "history", "social",
                   "economics", "시민", "법학", "철학"],
            # 29: 의료/간호
            "의료/간호": ["의료", "간호", "환자", "병원", "의학", "healthcare", "nursing",
                       "medical", "임상", "진료", "치료", "약학", "헬스케어"],
            # 31: 교육(교수·학습)
            "교육(교수·학습)": ["교수법", "교사", "수업", "교육학", "pedagogy",
                            "teaching", "강의", "교수설계", "커리큘럼"],
            # 32: 서비스/고객응대
            "서비스/고객응대": ["cs", "고객응대", "친절", "서비스", "customer service",
                           "고객만족", "상담", "클레임", "접객", "hospitality"],
        }

        # 키워드 매칭으로 도메인 추론
        for domain, keywords in domain_keywords.items():
            if any(kw in combined for kw in keywords):
                return domain

        return None

    @staticmethod
    def _infer_learner_role(context: dict) -> Optional[str]:
        """대상 정보에서 학습자 직업/역할 추론 (Context Matrix 13-16번)

        CSV 항목 정확히 반영:
        13: 학생, 14: 직장인(사무/관리), 15: 전문직(의료/법률/기술), 16: 예비 교사/교사
        """
        target = context.get("target_audience", "").lower()

        # 역할별 키워드 매핑 (우선순위: 구체적 → 일반적)
        role_keywords = {
            # 16: 예비 교사/교사 - 가장 구체적이므로 먼저 검사
            "예비 교사/교사": [
                "교사", "교원", "예비교사", "예비 교사", "임용", "교생",
                "교수", "강사", "튜터", "trainer", "instructor", "teacher",
                "teaching", "교육자", "훈련가"
            ],
            # 15: 전문직(의료/법률/기술) - 구체적 직종 명시
            "전문직(의료/법률/기술)": [
                # 의료
                "의사", "간호사", "약사", "치과의사", "한의사", "물리치료사",
                "의료인", "임상", "레지던트", "인턴", "nurse", "doctor",
                # 법률
                "변호사", "법무사", "검사", "판사", "lawyer", "attorney",
                "법률", "법조인",
                # 기술/엔지니어
                "엔지니어", "engineer", "기술자", "연구원", "박사", "석사",
                "전문가", "specialist", "컨설턴트", "consultant",
                # 기타 전문직
                "회계사", "세무사", "건축사", "감정평가사", "공인중개사"
            ],
            # 14: 직장인(사무/관리) - 일반 사무직/관리직
            "직장인(사무/관리)": [
                # 직급
                "사원", "대리", "과장", "차장", "부장", "팀장", "매니저",
                "manager", "supervisor", "리더", "실장", "임원", "이사",
                # 일반 직장인
                "직장인", "회사원", "직원", "사무직", "관리자", "관리직",
                "office", "worker", "employee", "staff",
                # 신입/경력
                "신입사원", "신입", "경력직", "이직자"
            ],
            # 13: 학생 - 가장 일반적이므로 마지막
            "학생": [
                "학생", "대학생", "학부생", "대학원생", "석사생", "박사생",
                "고등학생", "중학생", "초등학생", "student", "learner",
                "수강생", "교육생", "훈련생", "trainee", "연수생"
            ],
        }

        # 우선순위대로 키워드 매칭
        for role, keywords in role_keywords.items():
            if any(kw in target for kw in keywords):
                return role

        return None

    @staticmethod
    def _infer_class_size(context: dict) -> Optional[str]:
        """컨텍스트에서 학습자 규모 추론 (Context Matrix 40-42번)

        CSV 항목 정확히 반영:
        40: 소규모(1-10명), 41: 중규모(10-30명), 42: 대규모(30명 이상)
        """
        # 다양한 소스에서 규모 정보 수집
        sources = [
            context.get("class_size", ""),
            context.get("learner_count", ""),
            context.get("group_size", ""),
            context.get("target_audience", ""),
            str(context.get("constraints", "")),
        ]
        combined = " ".join(str(s) for s in sources).lower()

        # 숫자 패턴 추출 시도
        import re
        number_match = re.search(r"(\d+)\s*명", combined)
        if number_match:
            count = int(number_match.group(1))
            if count <= 10:
                return "소규모(1-10명)"
            elif count <= 30:
                return "중규모(10-30명)"
            else:
                return "대규모(30명 이상)"

        # 키워드 기반 추론
        size_keywords = {
            "소규모(1-10명)": [
                "소규모", "소수", "1:1", "일대일", "개별", "튜터링",
                "멘토링", "코칭", "5명", "3명", "개인지도"
            ],
            "중규모(10-30명)": [
                "중규모", "소집단", "팀", "그룹", "15명", "20명", "25명",
                "한 반", "학급", "분반"
            ],
            "대규모(30명 이상)": [
                "대규모", "대집단", "강의", "50명", "100명", "전사",
                "전체", "대형", "다수", "대인원"
            ],
        }

        for size, keywords in size_keywords.items():
            if any(kw in combined for kw in keywords):
                return size

        return None

    @staticmethod
    def _infer_tech_environment(context: dict) -> Optional[str]:
        """컨텍스트에서 기술 환경 추론 (Context Matrix 46-48번)

        CSV 항목 정확히 반영:
        46: 디지털 기기 제공, 47: 개인 기기 지참(BYOD), 48: 제한적 기술 환경
        """
        # 다양한 소스에서 기술 환경 정보 수집
        sources = [
            context.get("tech_environment", ""),
            context.get("learning_environment", ""),
            context.get("technology", ""),
            str(context.get("constraints", "")),
            str(context.get("resources", "")),
        ]
        combined = " ".join(str(s) for s in sources).lower()

        # 기술 환경 키워드 매핑
        tech_keywords = {
            # 46: 디지털 기기 제공 - 기관에서 기기 제공
            "디지털 기기 제공": [
                "기기 제공", "노트북 제공", "태블릿 제공", "pc 제공",
                "컴퓨터실", "전산실", "it 인프라", "디지털 교실",
                "스마트 교실", "첨단", "멀티미디어실"
            ],
            # 47: 개인 기기 지참(BYOD)
            "개인 기기 지참(BYOD)": [
                "byod", "개인 기기", "개인 노트북", "본인 기기",
                "자기 기기", "개인 스마트폰", "모바일 학습"
            ],
            # 48: 제한적 기술 환경
            "제한적 기술 환경": [
                "제한적", "저기술", "오프라인 전용", "인터넷 불가",
                "네트워크 제한", "기술 제약", "인쇄물", "종이 자료",
                "칠판", "화이트보드", "아날로그", "비디지털"
            ],
        }

        for env, keywords in tech_keywords.items():
            if any(kw in combined for kw in keywords):
                return env

        return None
