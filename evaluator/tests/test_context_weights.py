
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from isd_evaluator.metrics.context_weights import ContextWeightAdjuster
from isd_evaluator.models import ADDIEPhase
from isd_evaluator.rubrics.addie_definitions import DEFAULT_PHASE_WEIGHTS

class TestContextWeightAdjuster(unittest.TestCase):
    
    def test_issue_2_offline_delivery(self):
        """Issue #2: 오프라인 전달 방식 가중치 테스트"""
        scenario = {
            "context": {
                "learning_environment": "오프라인(교실 수업)",
                "target_audience": "일반인",
                "duration": "1일"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()
        
        # 기본 가중치와 비교
        # 오프라인: Implementation +0.10, Design +0.05
        # 기본: Implementation 0.15, Design 0.20
        # 예상: Implementation 증가, Design 증가 (relative to base, normalized)
        
        # Check raw logic (before normalization triggers significantly)
        # However, since we don't have access to interal raw weights easily without mocking, 
        # we check if Implementation weight is higher than Development which is lowered (-0.10)
        
        print(f"\n[Test Issue #2] Offline Weights: {weights}")
        
        # Implementation should be significantly weighted
        self.assertGreater(weights[ADDIEPhase.IMPLEMENTATION], DEFAULT_PHASE_WEIGHTS[ADDIEPhase.IMPLEMENTATION] * 0.9)
        
        # Development should be lower than Implementation in this context
        self.assertLess(weights[ADDIEPhase.DEVELOPMENT], weights[ADDIEPhase.IMPLEMENTATION])

    def test_issue_3_learner_elementary(self):
        """Issue #3: 초등학생 대상 가중치 테스트 (Dual-Track)"""
        scenario = {
            "context": {
                "target_audience": "초등학교 5학년",
                "learning_environment": "교실",
                "duration": "40분"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()
        
        print(f"\n[Test Issue #3] Elementary Target Weights: {weights}")
        
        # 초등: Development +0.10 (Visualization/Interest)
        # Development should be higher than Analysis (which is -0.05)
        self.assertGreater(weights[ADDIEPhase.DEVELOPMENT], weights[ADDIEPhase.ANALYSIS])

    def test_issue_3_learner_adult_expert(self):
        """Issue #3: 성인/전문가 대상 가중치 테스트 (Dual-Track)"""
        scenario = {
            "context": {
                "target_audience": "IT 기업 10년차 개발자",
                "prior_knowledge": "고급",
                "learning_environment": "온라인",
                "duration": "4주"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #3] Adult/Expert Target Weights: {weights}")

        # 성인/고급: Analysis +0.10, Evaluation +0.10
        # Analysis should be higher than Development (which is -0.10)
        self.assertGreater(weights[ADDIEPhase.ANALYSIS], weights[ADDIEPhase.DEVELOPMENT])

    # ========== Issue #16: 학습자 직업/역할 (Items 13-16) ==========

    def test_issue_16_student_role(self):
        """Issue #16: 학생 역할 가중치 테스트 (Item 13)"""
        scenario = {
            "context": {
                "target_audience": "컴퓨터공학과 대학생",
                "learning_environment": "온라인",
                "duration": "1학기"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #16] Student Role Weights: {weights}")
        print(f"  - learner_role: {adjuster.context_profile.learner_role}")

        # 학생: Design +0.05, Development +0.05
        # Design과 Development가 Analysis보다 높아야 함
        self.assertEqual(adjuster.context_profile.learner_role, "학생")
        self.assertGreater(weights[ADDIEPhase.DESIGN], weights[ADDIEPhase.ANALYSIS])
        self.assertGreater(weights[ADDIEPhase.DEVELOPMENT], weights[ADDIEPhase.ANALYSIS])

    def test_issue_16_office_worker_role(self):
        """Issue #16: 직장인 역할 가중치 테스트 (Item 14)"""
        scenario = {
            "context": {
                "target_audience": "마케팅팀 대리급 직원",
                "learning_environment": "블렌디드",
                "duration": "2주"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #16] Office Worker Role Weights: {weights}")
        print(f"  - learner_role: {adjuster.context_profile.learner_role}")

        # 직장인: Implementation +0.10 (현업 적용성)
        # Implementation이 Development보다 높아야 함
        self.assertEqual(adjuster.context_profile.learner_role, "직장인(사무/관리)")
        self.assertGreater(weights[ADDIEPhase.IMPLEMENTATION], weights[ADDIEPhase.DEVELOPMENT])

    def test_issue_16_professional_role(self):
        """Issue #16: 전문직 역할 가중치 테스트 (Item 15)"""
        scenario = {
            "context": {
                "target_audience": "대학병원 간호사",
                "learning_environment": "오프라인",  # 시뮬레이션 대신 오프라인 사용 (복합 효과 제거)
                "duration": "3일"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #16] Professional Role Weights: {weights}")
        print(f"  - learner_role: {adjuster.context_profile.learner_role}")

        # 전문직: Design +0.05, Evaluation +0.05 (사례 중심 + 정확성)
        self.assertEqual(adjuster.context_profile.learner_role, "전문직(의료/법률/기술)")
        # Design이 Development보다 높아야 함 (전문직 효과)
        self.assertGreater(weights[ADDIEPhase.DESIGN], weights[ADDIEPhase.DEVELOPMENT])

    def test_issue_16_teacher_role(self):
        """Issue #16: 교사 역할 가중치 테스트 (Item 16)"""
        scenario = {
            "context": {
                "target_audience": "임용고시 준비 예비교사",
                "learning_environment": "오프라인",
                "duration": "1학기"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #16] Teacher Role Weights: {weights}")
        print(f"  - learner_role: {adjuster.context_profile.learner_role}")

        # 예비교사/교사: Design +0.10 (교수설계 역량 모델링)
        # Design이 가장 높거나 상위권이어야 함
        self.assertEqual(adjuster.context_profile.learner_role, "예비 교사/교사")
        self.assertGreater(weights[ADDIEPhase.DESIGN], weights[ADDIEPhase.DEVELOPMENT])

    def test_issue_16_role_inference_keywords(self):
        """Issue #16: 역할 추론 키워드 테스트"""
        test_cases = [
            # (target_audience, expected_role)
            ("신입사원 교육 대상자", "직장인(사무/관리)"),
            ("IT 엔지니어", "전문직(의료/법률/기술)"),
            ("변호사 대상 연수", "전문직(의료/법률/기술)"),
            ("중학교 교사", "예비 교사/교사"),
            ("대학교 강사진", "예비 교사/교사"),
            ("고등학생", "학생"),
            ("대학원생 연구자", "학생"),
        ]

        for target, expected_role in test_cases:
            scenario = {"context": {"target_audience": target}}
            adjuster = ContextWeightAdjuster.from_scenario(scenario)
            actual_role = adjuster.context_profile.learner_role

            print(f"\n[Test Issue #16] '{target}' -> {actual_role}")
            self.assertEqual(
                actual_role, expected_role,
                f"'{target}'에서 '{expected_role}'를 기대했으나 '{actual_role}' 반환"
            )

    # ========== Issue #16 추가: 학습자 규모 (Items 40-42) ==========

    def test_issue_16_small_class_size(self):
        """Issue #16: 소규모 학습자 가중치 테스트 (Item 40)"""
        scenario = {
            "context": {
                "target_audience": "5명의 임원진",
                "learning_environment": "오프라인",
                "duration": "2일"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #16] Small Class Weights: {weights}")
        print(f"  - class_size: {adjuster.context_profile.class_size}")

        # 소규모: Evaluation +0.10 (개별 피드백)
        self.assertEqual(adjuster.context_profile.class_size, "소규모(1-10명)")
        self.assertGreater(weights[ADDIEPhase.EVALUATION], weights[ADDIEPhase.DEVELOPMENT])

    def test_issue_16_large_class_size(self):
        """Issue #16: 대규모 학습자 가중치 테스트 (Item 42)"""
        scenario = {
            "context": {
                "target_audience": "전사 직원 100명",
                "learning_environment": "온라인",
                "duration": "1주"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #16] Large Class Weights: {weights}")
        print(f"  - class_size: {adjuster.context_profile.class_size}")

        # 대규모: Development +0.10 (표준화 자료)
        self.assertEqual(adjuster.context_profile.class_size, "대규모(30명 이상)")
        self.assertGreater(weights[ADDIEPhase.DEVELOPMENT], weights[ADDIEPhase.EVALUATION])

    # ========== Issue #16 추가: 기술 환경 (Items 46-48) ==========

    def test_issue_16_digital_tech_environment(self):
        """Issue #16: 디지털 기기 제공 가중치 테스트 (Item 46)"""
        scenario = {
            "context": {
                "target_audience": "대학생",
                "learning_environment": "컴퓨터실",
                "duration": "3시간"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #16] Digital Tech Weights: {weights}")
        print(f"  - tech_environment: {adjuster.context_profile.tech_environment}")

        # 디지털 기기 제공: Development +0.10 (멀티미디어)
        self.assertEqual(adjuster.context_profile.tech_environment, "디지털 기기 제공")
        self.assertGreater(weights[ADDIEPhase.DEVELOPMENT], weights[ADDIEPhase.ANALYSIS])

    def test_issue_16_limited_tech_environment(self):
        """Issue #16: 제한적 기술 환경 가중치 테스트 (Item 48)"""
        scenario = {
            "context": {
                "target_audience": "농촌 지역 주민",
                "learning_environment": "오프라인 전용 마을회관",
                "duration": "4시간"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #16] Limited Tech Weights: {weights}")
        print(f"  - tech_environment: {adjuster.context_profile.tech_environment}")

        # 제한적 기술 환경: Development +0.10, Implementation +0.10
        self.assertEqual(adjuster.context_profile.tech_environment, "제한적 기술 환경")
        self.assertGreater(weights[ADDIEPhase.IMPLEMENTATION], weights[ADDIEPhase.DESIGN])

    def test_issue_16_class_size_inference(self):
        """Issue #16: 학습자 규모 추론 테스트"""
        test_cases = [
            # (context, expected_size)
            ({"target_audience": "3명의 팀원"}, "소규모(1-10명)"),
            ({"target_audience": "20명의 신입사원"}, "중규모(10-30명)"),
            ({"target_audience": "50명의 영업사원"}, "대규모(30명 이상)"),
            ({"class_size": "소규모"}, "소규모(1-10명)"),
            ({"target_audience": "1:1 코칭 대상"}, "소규모(1-10명)"),
        ]

        for context, expected_size in test_cases:
            scenario = {"context": context}
            adjuster = ContextWeightAdjuster.from_scenario(scenario)
            actual_size = adjuster.context_profile.class_size

            print(f"\n[Test Issue #16] {context} -> {actual_size}")
            self.assertEqual(
                actual_size, expected_size,
                f"{context}에서 '{expected_size}'를 기대했으나 '{actual_size}' 반환"
            )

    # ========== Issue #4: 교육 도메인 (Items 23-32) ==========

    def test_issue_4_domain_language(self):
        """Issue #4: 언어 도메인 가중치 테스트 (Item 23)"""
        scenario = {
            "context": {
                "target_audience": "대학생",  # 직장인 대신 대학생으로 변경 (복합 효과 최소화)
                "topic": "비즈니스 영어 회화",
                "learning_environment": "교실",  # 온라인 대신 교실로 변경
                "duration": "1학기"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Language Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 언어: Implementation +0.10 (말하기/듣기 실습), Design +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "언어")
        # Implementation이 Development보다 높아야 함 (언어 도메인 효과)
        self.assertGreater(weights[ADDIEPhase.IMPLEMENTATION], weights[ADDIEPhase.DEVELOPMENT])

    def test_issue_4_domain_math(self):
        """Issue #4: 수학 도메인 가중치 테스트 (Item 24)"""
        scenario = {
            "context": {
                "target_audience": "고등학생",
                "topic": "미적분 기초",
                "subject": "수학",
                "learning_environment": "교실",
                "duration": "1학기"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Math Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 수학: Design +0.10 (개념 구조화), Development +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "수학")
        self.assertGreater(weights[ADDIEPhase.DESIGN], weights[ADDIEPhase.IMPLEMENTATION])

    def test_issue_4_domain_science(self):
        """Issue #4: 과학 도메인 가중치 테스트 (Item 25)"""
        scenario = {
            "context": {
                "target_audience": "중학생",
                "topic": "화학 실험 기초",
                "learning_environment": "과학실",
                "duration": "1학기"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Science Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 과학: Development +0.10 (실험/시각화), Design +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "과학")
        self.assertGreater(weights[ADDIEPhase.DEVELOPMENT], weights[ADDIEPhase.EVALUATION])

    def test_issue_4_domain_social(self):
        """Issue #4: 사회 도메인 가중치 테스트 (Item 26)"""
        scenario = {
            "context": {
                "target_audience": "고등학생",
                "topic": "현대 한국사",
                "subject": "역사",
                "learning_environment": "교실",
                "duration": "1학기"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Social Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 사회: Analysis +0.05, Design +0.05 (토론/협력 활동)
        self.assertEqual(adjuster.context_profile.education_domain, "사회")

    def test_issue_4_domain_software_dev(self):
        """Issue #4: 개발(Software/IT) 도메인 가중치 테스트 (Item 27)"""
        scenario = {
            "context": {
                "target_audience": "신입 개발자",
                "topic": "파이썬 프로그래밍 기초",
                "learning_environment": "온라인",
                "duration": "4주"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Software Dev Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 개발: Implementation +0.10 (실습 환경), Development +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "개발(Software/IT)")
        self.assertGreater(weights[ADDIEPhase.IMPLEMENTATION], weights[ADDIEPhase.ANALYSIS])

    def test_issue_4_domain_ai(self):
        """Issue #4: AI 도메인 가중치 테스트 (Item 28)"""
        scenario = {
            "context": {
                "target_audience": "데이터 분석가",
                "topic": "머신러닝 기초",
                "objectives": ["딥러닝 모델 구축", "AI 활용"],
                "learning_environment": "온라인",
                "duration": "8주"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] AI Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # AI: Implementation +0.10 (실습 환경), Development +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "AI")
        self.assertGreater(weights[ADDIEPhase.IMPLEMENTATION], weights[ADDIEPhase.ANALYSIS])

    def test_issue_4_domain_medical(self):
        """Issue #4: 의료/간호 도메인 가중치 테스트 (Item 29)"""
        scenario = {
            "context": {
                "target_audience": "대학생",  # 전문직 효과 제거를 위해 대학생으로 변경
                "topic": "간호학 임상 실습 개론",
                "learning_environment": "교실",  # 시뮬레이션 대신 교실로 변경 (복합 효과 최소화)
                "duration": "1학기"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Medical Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 의료/간호: Evaluation +0.10 (절차 정확성), Development +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "의료/간호")
        # Evaluation이 Analysis보다 높아야 함 (의료 도메인 효과: Analysis -0.05, Evaluation +0.10)
        self.assertGreater(weights[ADDIEPhase.EVALUATION], weights[ADDIEPhase.ANALYSIS])

    def test_issue_4_domain_business(self):
        """Issue #4: 경영/HR 도메인 가중치 테스트 (Item 30)"""
        scenario = {
            "context": {
                "target_audience": "중간 관리자",
                "topic": "리더십 역량 개발",
                "learning_environment": "블렌디드",
                "duration": "3개월"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Business Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 경영/HR: Analysis +0.05 (요구분석), Evaluation +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "경영/HR/경영지원")

    def test_issue_4_domain_education(self):
        """Issue #4: 교육(교수·학습) 도메인 가중치 테스트 (Item 31)"""
        scenario = {
            "context": {
                "target_audience": "예비교사",
                "topic": "효과적인 수업 교수법",
                "learning_environment": "대면",
                "duration": "1학기"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Education Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 교육: Design +0.10 (교수법 설계), Evaluation +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "교육(교수·학습)")
        self.assertGreater(weights[ADDIEPhase.DESIGN], weights[ADDIEPhase.DEVELOPMENT])

    def test_issue_4_domain_service(self):
        """Issue #4: 서비스/고객응대 도메인 가중치 테스트 (Item 32)"""
        scenario = {
            "context": {
                "target_audience": "콜센터 상담원",
                "topic": "CS 고객응대 교육",
                "learning_environment": "오프라인",
                "duration": "2일"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        weights = adjuster.get_adjusted_weights()

        print(f"\n[Test Issue #4] Service Domain Weights: {weights}")
        print(f"  - education_domain: {adjuster.context_profile.education_domain}")

        # 서비스: Implementation +0.10 (롤플레이), Evaluation +0.05
        self.assertEqual(adjuster.context_profile.education_domain, "서비스/고객응대")
        self.assertGreater(weights[ADDIEPhase.IMPLEMENTATION], weights[ADDIEPhase.ANALYSIS])

    def test_issue_4_domain_inference_keywords(self):
        """Issue #4: 도메인 추론 키워드 테스트"""
        test_cases = [
            # (context, expected_domain)
            ({"topic": "영어 문법 기초"}, "언어"),
            ({"topic": "통계학 입문"}, "수학"),
            ({"topic": "물리 실험"}, "과학"),
            ({"topic": "세계사와 지리"}, "사회"),
            ({"topic": "자바 프로그래밍"}, "개발(Software/IT)"),
            ({"topic": "딥러닝 모델 학습"}, "AI"),
            ({"topic": "임상 간호 실습"}, "의료/간호"),
            ({"topic": "마케팅 전략 수립"}, "경영/HR/경영지원"),
            ({"topic": "효과적인 강의 기법"}, "교육(교수·학습)"),
            ({"topic": "고객 클레임 처리"}, "서비스/고객응대"),
        ]

        for context, expected_domain in test_cases:
            scenario = {"context": context}
            adjuster = ContextWeightAdjuster.from_scenario(scenario)
            actual_domain = adjuster.context_profile.education_domain

            print(f"\n[Test Issue #4] {context} -> {actual_domain}")
            self.assertEqual(
                actual_domain, expected_domain,
                f"{context}에서 '{expected_domain}'를 기대했으나 '{actual_domain}' 반환"
            )

    def test_issue_4_domain_priority_business_over_dev(self):
        """Issue #4: 경영 도메인이 '개발'보다 우선순위 높은지 테스트 (리더십 개발 등)"""
        # "리더십 개발"은 경영/HR 도메인이어야 함 (개발(IT)이 아님)
        scenario = {
            "context": {
                "target_audience": "팀장급 관리자",
                "topic": "리더십 개발 과정",
                "learning_environment": "오프라인",
                "duration": "2일"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)

        print(f"\n[Test Issue #4] Leadership Development -> {adjuster.context_profile.education_domain}")

        # "리더십 개발"은 경영/HR 도메인으로 분류되어야 함
        self.assertEqual(adjuster.context_profile.education_domain, "경영/HR/경영지원")

    def test_issue_4_domain_priority_ai_over_dev(self):
        """Issue #4: AI 도메인이 개발(IT)보다 우선순위 높은지 테스트"""
        # "AI 개발"은 AI 도메인이어야 함 (개발(IT)이 아님)
        scenario = {
            "context": {
                "target_audience": "개발자",
                "topic": "AI 모델 개발 및 배포",
                "learning_environment": "온라인",
                "duration": "4주"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)

        print(f"\n[Test Issue #4] AI Development -> {adjuster.context_profile.education_domain}")

        # "AI 모델 개발"은 AI 도메인으로 분류되어야 함
        self.assertEqual(adjuster.context_profile.education_domain, "AI")

if __name__ == '__main__':
    unittest.main()
