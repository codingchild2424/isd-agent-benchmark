"""Marp Exporter 단위 테스트"""
import os
import sys
import tempfile
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.utils.marp_exporter import to_marp_markdown, export_to_file


def test_to_marp_markdown_basic():
    """기본 슬라이드 변환 테스트"""
    print("Testing basic Marp markdown conversion...", end=" ")

    slides = [
        {
            "slide_number": 1,
            "title": "테스트 슬라이드",
            "bullet_points": ["포인트 1", "포인트 2"],
            "speaker_notes": "발표자 메모입니다."
        }
    ]

    result = to_marp_markdown(slides, title="테스트 프레젠테이션")

    # YAML Front Matter 확인
    assert "marp: true" in result
    assert "title: 테스트 프레젠테이션" in result

    # 슬라이드 제목 확인
    assert "# 테스트 슬라이드" in result

    # Bullet points 확인
    assert "- 포인트 1" in result
    assert "- 포인트 2" in result

    # 발표자 노트 확인
    assert "<!-- 발표자 노트: 발표자 메모입니다. -->" in result

    print("PASS")


def test_to_marp_markdown_multiple_slides():
    """다중 슬라이드 변환 테스트"""
    print("Testing multiple slides conversion...", end=" ")

    slides = [
        {"slide_number": 1, "title": "슬라이드 1", "bullet_points": ["A"]},
        {"slide_number": 2, "title": "슬라이드 2", "bullet_points": ["B"]},
        {"slide_number": 3, "title": "슬라이드 3", "bullet_points": ["C"]},
    ]

    result = to_marp_markdown(slides)

    # 슬라이드 구분선 확인 (Front Matter 포함하여 최소 2개 이상)
    # Front Matter ---가 2개 + 슬라이드 구분선 2개 = 총 4개 이상
    total_separators = result.count("---")
    assert total_separators >= 2, f"Expected at least 2 separators, got {total_separators}"

    # 모든 슬라이드 제목 확인
    assert "# 슬라이드 1" in result
    assert "# 슬라이드 2" in result
    assert "# 슬라이드 3" in result

    print("PASS")


def test_to_marp_markdown_with_visual_suggestion():
    """시각 자료 제안 포함 테스트"""
    print("Testing visual suggestion inclusion...", end=" ")

    slides = [
        {
            "slide_number": 1,
            "title": "차트 슬라이드",
            "bullet_points": ["데이터 설명"],
            "visual_suggestion": "막대 그래프 삽입"
        }
    ]

    result = to_marp_markdown(slides)

    assert "<!-- 시각 자료: 막대 그래프 삽입 -->" in result

    print("PASS")


def test_to_marp_markdown_empty_slides():
    """빈 슬라이드 리스트 테스트"""
    print("Testing empty slides list...", end=" ")

    result = to_marp_markdown([])

    # Front Matter만 있어야 함
    assert "marp: true" in result
    assert "---" in result

    print("PASS")


def test_to_marp_markdown_theme_options():
    """테마 옵션 테스트"""
    print("Testing theme options...", end=" ")

    slides = [{"slide_number": 1, "title": "Test", "bullet_points": []}]

    # gaia 테마
    result_gaia = to_marp_markdown(slides, theme="gaia")
    assert "theme: gaia" in result_gaia

    # uncover 테마
    result_uncover = to_marp_markdown(slides, theme="uncover")
    assert "theme: uncover" in result_uncover

    # paginate false
    result_no_paginate = to_marp_markdown(slides, paginate=False)
    assert "paginate: false" in result_no_paginate

    print("PASS")


def test_export_to_file():
    """파일 저장 테스트"""
    print("Testing file export...", end=" ")

    slides = [
        {"slide_number": 1, "title": "저장 테스트", "bullet_points": ["테스트 항목"]}
    ]

    # 임시 파일 생성
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_slides.md")

        result_path = export_to_file(slides, output_path, title="저장 테스트")

        # 파일 존재 확인
        assert os.path.exists(result_path)

        # 파일 내용 확인
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# 저장 테스트" in content
        assert "- 테스트 항목" in content

    print("PASS")


def test_export_to_file_nested_directory():
    """중첩 디렉토리 생성 테스트"""
    print("Testing nested directory creation...", end=" ")

    slides = [{"slide_number": 1, "title": "Test", "bullet_points": []}]

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = os.path.join(tmpdir, "a", "b", "c", "slides.md")

        result_path = export_to_file(slides, nested_path)

        assert os.path.exists(result_path)
        assert os.path.isfile(result_path)

    print("PASS")


def test_korean_content():
    """한글 콘텐츠 처리 테스트"""
    print("Testing Korean content handling...", end=" ")

    slides = [
        {
            "slide_number": 1,
            "title": "학습 목표 소개",
            "bullet_points": [
                "회사 조직 구조를 이해한다",
                "사내 시스템 사용법을 익힌다",
                "핵심 가치를 설명할 수 있다"
            ],
            "speaker_notes": "학습자들에게 오늘의 목표를 명확히 전달합니다."
        }
    ]

    result = to_marp_markdown(slides, title="신입사원 교육")

    assert "학습 목표 소개" in result
    assert "회사 조직 구조를 이해한다" in result
    assert "신입사원 교육" in result

    print("PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Marp Exporter 단위 테스트")
    print("=" * 60)

    tests = [
        test_to_marp_markdown_basic,
        test_to_marp_markdown_multiple_slides,
        test_to_marp_markdown_with_visual_suggestion,
        test_to_marp_markdown_empty_slides,
        test_to_marp_markdown_theme_options,
        test_export_to_file,
        test_export_to_file_nested_directory,
        test_korean_content,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL - {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR - {e}")
            failed += 1

    print()
    print(f"결과: {passed}/{len(tests)} 테스트 통과")

    if failed > 0:
        sys.exit(1)
