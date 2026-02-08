"""
ADDIE 루브릭 파싱 로직 단위 테스트

테스트 대상: ADDIERubricEvaluator._parse_sub_item_result()
- 정상 JSON 응답 파싱
- 깨진 JSON 처리
- 형식 오류 처리
- 필드 누락 처리
"""

import pytest
from unittest.mock import MagicMock


class TestParseSubItemResult:
    """_parse_sub_item_result 메서드 테스트"""

    # ========== 정상 케이스 ==========

    def test_parse_valid_json_response(self, mock_evaluator, load_llm_response):
        """정상적인 JSON 응답 파싱 테스트"""
        response = load_llm_response("analysis/valid_response.json")
        sub_item_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        # 모든 항목이 파싱되었는지 확인
        assert len(sub_scores) == 10
        assert len(sub_reasoning) == 10

        # 점수 범위 확인
        for sid, score in sub_scores.items():
            assert 0.0 <= score <= 10.0

        # 특정 점수 확인
        assert sub_scores[1] == 8.5
        assert sub_scores[5] == 9.0
        assert sub_scores[4] == 6.5

    def test_parse_valid_json_with_markdown_fence(self, mock_evaluator):
        """Markdown 코드 펜스로 감싸진 JSON 파싱"""
        response = '''```json
{
  "sub_scores": {"1": 8.0, "2": 7.5, "3": 6.0},
  "sub_reasoning": {"1": "양호", "2": "적절", "3": "보통"},
  "missing_elements": [],
  "weak_areas": []
}
```'''
        sub_item_ids = [1, 2, 3]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        assert sub_scores[1] == 8.0
        assert sub_scores[2] == 7.5
        assert sub_scores[3] == 6.0

    def test_parse_valid_json_without_fence(self, mock_evaluator):
        """코드 펜스 없는 순수 JSON 파싱"""
        response = '''{
  "sub_scores": {"1": 9.0, "2": 8.5},
  "sub_reasoning": {"1": "우수", "2": "양호"},
  "missing_elements": [],
  "weak_areas": []
}'''
        sub_item_ids = [1, 2]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        assert sub_scores[1] == 9.0
        assert sub_scores[2] == 8.5

    def test_parse_reasoning_extracted(self, mock_evaluator, load_llm_response):
        """reasoning 필드가 올바르게 추출되는지 확인"""
        response = load_llm_response("analysis/valid_response.json")
        sub_item_ids = [1, 2, 3]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        # reasoning이 존재하고 내용이 있는지 확인
        assert 1 in sub_reasoning
        assert len(sub_reasoning[1]) > 0
        assert "문제 확인" in sub_reasoning[1]

    # ========== 에러 케이스 ==========

    def test_parse_broken_json_returns_default(self, mock_evaluator, load_llm_response):
        """깨진 JSON 응답 시 기본값 반환"""
        response = load_llm_response("analysis/broken_json.txt")
        sub_item_ids = [1, 2, 3]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        # 기본값 5.0이 반환되어야 함
        for sid in sub_item_ids:
            assert sub_scores[sid] == 5.0

    def test_parse_malformed_structure_returns_default(self, mock_evaluator, load_llm_response):
        """형식이 잘못된 JSON 구조 처리 (sub_scores 키 없음)"""
        response = load_llm_response("analysis/malformed_structure.json")
        sub_item_ids = [1, 2, 3]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        # sub_scores 키가 없으므로 기본값 반환
        for sid in sub_item_ids:
            assert sub_scores[sid] == 5.0

    def test_parse_missing_fields_partial_parse(self, mock_evaluator, load_llm_response):
        """일부 필드 누락 시 부분 파싱"""
        response = load_llm_response("analysis/missing_fields.json")
        sub_item_ids = [1, 2, 3, 4, 5]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        # 존재하는 항목은 파싱되고, 없는 항목은 기본값
        assert sub_scores[1] == 8.5
        assert sub_scores[2] == 7.0
        assert sub_scores[3] == 5.0  # 기본값
        assert sub_scores[4] == 5.0  # 기본값
        assert sub_scores[5] == 5.0  # 기본값

    def test_parse_empty_response(self, mock_evaluator):
        """빈 응답 처리"""
        response = ""
        sub_item_ids = [1, 2]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        # 모든 항목에 기본값
        for sid in sub_item_ids:
            assert sub_scores[sid] == 5.0

    def test_parse_none_like_response(self, mock_evaluator):
        """None과 유사한 응답 처리"""
        response = "평가를 수행할 수 없습니다."
        sub_item_ids = [1, 2, 3]

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_item_ids
        )

        # 모든 항목에 기본값
        for sid in sub_item_ids:
            assert sub_scores[sid] == 5.0

    # ========== 경계 케이스 ==========

    def test_parse_score_clamping_high(self, mock_evaluator):
        """높은 점수(>10) 범위 제한 테스트"""
        response = '''```json
{
  "sub_scores": {"1": 15.0, "2": 12.5, "3": 10.0},
  "sub_reasoning": {"1": "초과", "2": "초과", "3": "최대"},
  "missing_elements": [],
  "weak_areas": []
}
```'''
        sub_item_ids = [1, 2, 3]

        sub_scores, _, _, _ = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        # 10.0 초과 점수는 10.0으로 클램핑
        assert sub_scores[1] == 10.0
        assert sub_scores[2] == 10.0
        assert sub_scores[3] == 10.0

    def test_parse_score_clamping_low(self, mock_evaluator):
        """낮은 점수(<0) 범위 제한 테스트"""
        response = '''```json
{
  "sub_scores": {"1": -5.0, "2": -1.0, "3": 0.0},
  "sub_reasoning": {"1": "음수", "2": "음수", "3": "최소"},
  "missing_elements": [],
  "weak_areas": []
}
```'''
        sub_item_ids = [1, 2, 3]

        sub_scores, _, _, _ = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        # 0.0 미만 점수는 0.0으로 클램핑
        assert sub_scores[1] == 0.0
        assert sub_scores[2] == 0.0
        assert sub_scores[3] == 0.0

    def test_parse_placeholder_scores(self, mock_evaluator):
        """<0.0-10.0> 플레이스홀더 처리"""
        response = '''```json
{
  "sub_scores": {"1": <0.0-10.0>, "2": 7.5},
  "sub_reasoning": {"1": "평가", "2": "평가"},
  "missing_elements": [],
  "weak_areas": []
}
```'''
        sub_item_ids = [1, 2]

        sub_scores, _, _, _ = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        # 플레이스홀더는 5.0으로 대체되어야 함
        assert sub_scores[1] == 5.0
        assert sub_scores[2] == 7.5

    def test_parse_double_brace_cleanup(self, mock_evaluator):
        """}} -> } 정리 테스트"""
        response = '''```json
{
  "sub_scores": {"1": 8.0, "2": 7.5}}
}
```'''
        sub_item_ids = [1, 2]

        # 이중 중괄호가 있어도 파싱 시도
        sub_scores, _, _, _ = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        # 파싱 성공 또는 기본값
        assert 1 in sub_scores
        assert 2 in sub_scores

    def test_parse_string_scores_converted(self, mock_evaluator):
        """문자열 점수가 float로 변환되는지 확인"""
        response = '''```json
{
  "sub_scores": {"1": "8.0", "2": "7.5"},
  "sub_reasoning": {},
  "missing_elements": [],
  "weak_areas": []
}
```'''
        sub_item_ids = [1, 2]

        sub_scores, _, _, _ = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        # 문자열이 float로 변환되어야 함
        assert isinstance(sub_scores[1], float)
        assert sub_scores[1] == 8.0

    def test_parse_integer_keys_handled(self, mock_evaluator):
        """정수 키와 문자열 키 모두 처리"""
        response = '''```json
{
  "sub_scores": {1: 8.0, "2": 7.5},
  "sub_reasoning": {},
  "missing_elements": [],
  "weak_areas": []
}
```'''
        sub_item_ids = [1, 2]

        # JSON에서 정수 키는 유효하지 않지만, 파싱 시도
        sub_scores, _, _, _ = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        # 파싱 실패 시 기본값
        assert 1 in sub_scores
        assert 2 in sub_scores


class TestParseSubItemResultAllPhases:
    """모든 ADDIE 단계의 파싱 테스트"""

    @pytest.mark.parametrize("phase,sub_items", [
        ("analysis", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ("design", [11, 12, 13, 14, 15, 16, 17, 18]),
        ("development", [19, 20, 21, 22, 23]),
        ("implementation", [24, 25, 26, 27]),
        ("evaluation", [28, 29, 30, 31, 32, 33]),
    ])
    def test_parse_all_phases(self, mock_evaluator, load_llm_response, phase, sub_items):
        """모든 ADDIE 단계의 유효한 응답 파싱 테스트"""
        response = load_llm_response(f"{phase}/valid_response.json")

        sub_scores, sub_reasoning, missing, weak = mock_evaluator._parse_sub_item_result(
            response, sub_items
        )

        # 모든 항목이 파싱되었는지 확인
        assert len(sub_scores) == len(sub_items)

        # 각 항목의 점수가 유효 범위인지 확인
        for sid in sub_items:
            assert sid in sub_scores
            assert 0.0 <= sub_scores[sid] <= 10.0

    @pytest.mark.parametrize("phase,expected_count", [
        ("analysis", 10),
        ("design", 8),
        ("development", 5),
        ("implementation", 4),
        ("evaluation", 6),
    ])
    def test_phase_item_counts(self, mock_evaluator, load_llm_response, phase, expected_count):
        """각 단계별 항목 수 확인"""
        phase_items = {
            "analysis": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "design": [11, 12, 13, 14, 15, 16, 17, 18],
            "development": [19, 20, 21, 22, 23],
            "implementation": [24, 25, 26, 27],
            "evaluation": [28, 29, 30, 31, 32, 33],
        }

        response = load_llm_response(f"{phase}/valid_response.json")
        sub_items = phase_items[phase]

        sub_scores, _, _, _ = mock_evaluator._parse_sub_item_result(response, sub_items)

        assert len(sub_scores) == expected_count


class TestParseSubItemResultMissingElements:
    """missing_elements와 weak_areas 추출 테스트"""

    def test_extract_missing_elements(self, mock_evaluator, load_llm_response):
        """missing_elements 필드 추출"""
        response = load_llm_response("analysis/valid_response.json")
        sub_item_ids = [1, 2, 3]

        _, _, missing, _ = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        # missing_elements가 리스트로 반환되는지 확인
        assert isinstance(missing, list)

    def test_empty_missing_elements(self, mock_evaluator):
        """빈 missing_elements 처리"""
        response = '''```json
{
  "sub_scores": {"1": 8.0},
  "sub_reasoning": {},
  "missing_elements": [],
  "weak_areas": []
}
```'''
        sub_item_ids = [1]

        _, _, missing, weak = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        assert missing == []
        assert weak == []

    def test_multiple_missing_elements(self, mock_evaluator):
        """여러 개의 missing_elements 추출"""
        response = '''```json
{
  "sub_scores": {"1": 5.0, "2": 4.0},
  "sub_reasoning": {},
  "missing_elements": ["요소1", "요소2", "요소3"],
  "weak_areas": ["약점1", "약점2"]
}
```'''
        sub_item_ids = [1, 2]

        _, _, missing, weak = mock_evaluator._parse_sub_item_result(response, sub_item_ids)

        assert len(missing) == 3
        assert len(weak) == 2
