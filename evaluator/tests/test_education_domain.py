
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from isd_evaluator.metrics.context_weights import ContextWeightAdjuster
from isd_evaluator.models import ADDIEPhase

class TestEducationDomainWeights(unittest.TestCase):
    
    def test_domain_inference_language(self):
        """도메인 추론 테스트: 언어 (Item 23)"""
        scenario = {
            "context": {
                "topic": "비즈니스 영어 이메일 작성",
                "objectives": ["효과적인 영문 이메일 작성"]
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        self.assertEqual(adjuster.context_profile.education_domain, "언어")
        
        weights = adjuster.get_adjusted_weights()
        # 언어: Imp +0.10, Des +0.05
        # Implementation 비중이 가장 높아야 함 (실습 중심)
        self.assertGreater(weights[ADDIEPhase.IMPLEMENTATION], weights[ADDIEPhase.ANALYSIS])

    def test_domain_inference_it_dev(self):
        """도메인 추론 테스트: 개발(IT) (Item 27)"""
        scenario = {
            "context": {
                "topic": "파이썬 기초 프로그래밍",
                "skills_to_acquire": ["변수", "함수", "제어문 코딩"]
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        self.assertEqual(adjuster.context_profile.education_domain, "개발(Software/IT)")
        
        weights = adjuster.get_adjusted_weights()
        # 개발: Imp +0.10, Dev +0.05
        # Implementation과 Development가 높아야 함
        self.assertGreaterEqual(weights[ADDIEPhase.IMPLEMENTATION], 0.25)

    def test_domain_inference_medical(self):
        """도메인 추론 테스트: 의료/간호 (Item 29)"""
        scenario = {
            "context": {
                "topic": "응급실 환자 분류(Triage)",
                "target_audience": "신규 간호사"
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        self.assertEqual(adjuster.context_profile.education_domain, "의료/간호")
        
        weights = adjuster.get_adjusted_weights()
        # 의료: Eval +0.10, Dev +0.05
        # Evaluation 비중이 Analysis보다 높아야 함 (엄격한 절차 평가)
        self.assertGreater(weights[ADDIEPhase.EVALUATION], weights[ADDIEPhase.ANALYSIS])

    def test_domain_priority_business_over_it(self):
        """도메인 우선순위 테스트: 리더십 개발 (경영 vs 개발)"""
        # '개발'이라는 단어가 포함되어 있어도 '리더십'이 있으면 경영/HR로 분류되어야 함
        scenario = {
            "context": {
                "topic": "팀장 리더십 역량 개발 과정",
                "objectives": ["조직 관리 능력 향상"]
            }
        }
        adjuster = ContextWeightAdjuster.from_scenario(scenario)
        self.assertEqual(adjuster.context_profile.education_domain, "경영/HR/경영지원")

if __name__ == '__main__':
    unittest.main()
