"""
Test: _ensure_required_fields() fallback logic (#71)

Verify that default values are correctly applied when pilot_data_collection is missing.
"""

import unittest
from baseline.generator import BaselineGenerator


class TestEnsureRequiredFields(unittest.TestCase):
    """_ensure_required_fields 메서드 테스트"""

    def setUp(self):
        """테스트용 제너레이터 인스턴스 생성 (API 호출 없음)"""
        self.generator = BaselineGenerator.__new__(BaselineGenerator)

    def test_pilot_data_collection_missing(self):
        """[28] pilot_data_collection이 누락된 경우 기본값 적용"""
        # 누락된 케이스
        data = {
            "evaluation": {
                "quiz_items": [{"id": "Q-01", "question": "테스트 문항"}]
                # pilot_data_collection 누락
            }
        }

        result = self.generator._ensure_required_fields(data)

        # pilot_data_collection이 추가되었는지 확인
        self.assertIn("pilot_data_collection", result["evaluation"])
        pdc = result["evaluation"]["pilot_data_collection"]

        # 기본값 구조 검증
        self.assertIsNotNone(pdc)
        self.assertIn("title", pdc)
        self.assertIn("data_types", pdc)
        self.assertIn("collection_methods", pdc)

    def test_pilot_data_collection_null(self):
        """[28] pilot_data_collection이 null인 경우 기본값 적용"""
        data = {
            "evaluation": {
                "quiz_items": [],
                "pilot_data_collection": None  # null 케이스
            }
        }

        result = self.generator._ensure_required_fields(data)

        # null이 기본값으로 대체되었는지 확인
        self.assertIsNotNone(result["evaluation"]["pilot_data_collection"])
        self.assertIn("title", result["evaluation"]["pilot_data_collection"])

    def test_pilot_data_collection_empty_dict(self):
        """[28] pilot_data_collection이 빈 dict인 경우 기본값 적용"""
        data = {
            "evaluation": {
                "pilot_data_collection": {}  # 빈 dict
            }
        }

        result = self.generator._ensure_required_fields(data)

        # 빈 dict이 기본값으로 대체되었는지 확인
        pdc = result["evaluation"]["pilot_data_collection"]
        self.assertIn("title", pdc)
        self.assertIn("data_types", pdc)

    def test_pilot_data_collection_existing(self):
        """[28] pilot_data_collection이 이미 있는 경우 유지"""
        existing_pdc = {
            "title": "커스텀 파일럿 계획",
            "data_types": {"quantitative": [], "qualitative": []},
            "collection_methods": [{"method": "설문"}]
        }
        data = {
            "evaluation": {
                "pilot_data_collection": existing_pdc
            }
        }

        result = self.generator._ensure_required_fields(data)

        # 기존 값이 유지되는지 확인
        self.assertEqual(
            result["evaluation"]["pilot_data_collection"]["title"],
            "커스텀 파일럿 계획"
        )

    def test_evaluation_missing(self):
        """evaluation 섹션 자체가 없는 경우"""
        data = {
            "analysis": {},
            "design": {}
            # evaluation 누락
        }

        result = self.generator._ensure_required_fields(data)

        # evaluation 섹션이 생성되고 pilot_data_collection이 추가되는지 확인
        self.assertIn("evaluation", result)
        self.assertIn("pilot_data_collection", result["evaluation"])


class TestParseResponseWithFallback(unittest.TestCase):
    """_parse_response 메서드에서 fallback 적용 테스트"""

    def setUp(self):
        self.generator = BaselineGenerator.__new__(BaselineGenerator)

    def test_json_without_pilot_data_collection(self):
        """JSON 파싱 성공 후 pilot_data_collection fallback 적용"""
        json_content = '''```json
{
    "analysis": {},
    "design": {},
    "development": {},
    "implementation": {},
    "evaluation": {
        "quiz_items": []
    }
}
```'''

        result = self.generator._parse_response(json_content)

        # fallback이 적용되어 pilot_data_collection이 존재하는지 확인
        self.assertIn("pilot_data_collection", result["evaluation"])
        self.assertIsNotNone(result["evaluation"]["pilot_data_collection"])


if __name__ == "__main__":
    unittest.main()
