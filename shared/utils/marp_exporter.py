"""Marp Markdown 변환 유틸리티"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import subprocess
import tempfile
import shutil
import json
import os


def to_marp_markdown(
    slide_contents: List[Dict[str, Any]],
    title: str = "프레젠테이션",
    theme: str = "default",
    paginate: bool = True
) -> str:
    """
    슬라이드 콘텐츠를 Marp Markdown 형식으로 변환

    Args:
        slide_contents: 슬라이드 콘텐츠 리스트
        title: 프레젠테이션 제목
        theme: Marp 테마 (default, gaia, uncover)
        paginate: 페이지 번호 표시 여부

    Returns:
        Marp Markdown 문자열
    """
    # YAML Front Matter
    lines = [
        "---",
        "marp: true",
        f"theme: {theme}",
        f"paginate: {str(paginate).lower()}",
        f"title: {title}",
        "---",
        ""
    ]

    for slide in slide_contents:
        slide_num = slide.get("slide_number", 0)
        slide_title = slide.get("title", "")
        bullet_points = slide.get("bullet_points", [])
        speaker_notes = slide.get("speaker_notes", "")
        visual_suggestion = slide.get("visual_suggestion", "")

        # 슬라이드 제목
        lines.append(f"# {slide_title}")
        lines.append("")

        # Bullet points
        for point in bullet_points:
            lines.append(f"- {point}")

        if bullet_points:
            lines.append("")

        # 시각 자료 제안 (주석으로)
        if visual_suggestion:
            lines.append(f"<!-- 시각 자료: {visual_suggestion} -->")

        # 발표자 노트
        if speaker_notes:
            lines.append(f"<!-- 발표자 노트: {speaker_notes} -->")

        # 슬라이드 구분선
        lines.append("")
        lines.append("---")
        lines.append("")

    # 마지막 구분선 제거
    if lines[-3:] == ["", "---", ""]:
        lines = lines[:-3]

    return "\n".join(lines)


def export_to_file(
    slide_contents: List[Dict[str, Any]],
    output_path: str,
    title: str = "프레젠테이션",
    theme: str = "default"
) -> str:
    """
    슬라이드 콘텐츠를 Marp Markdown 파일로 저장

    Args:
        slide_contents: 슬라이드 콘텐츠 리스트
        output_path: 출력 파일 경로 (.md)
        title: 프레젠테이션 제목
        theme: Marp 테마

    Returns:
        저장된 파일 경로
    """
    markdown = to_marp_markdown(slide_contents, title, theme)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")

    return str(path)


def _find_chrome_path() -> Optional[str]:
    """Chrome/Chromium 실행 파일 경로 찾기"""
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    elif system == "Linux":
        paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
        ]
    elif system == "Windows":
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
    else:
        return None

    for path in paths:
        if Path(path).exists():
            return path

    return None


def convert_to_pdf(markdown_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Marp Markdown을 PDF로 변환 (marp-cli 필요)

    Args:
        markdown_path: 입력 Markdown 파일 경로
        output_path: 출력 PDF 파일 경로 (없으면 자동 생성)

    Returns:
        생성된 PDF 파일 경로 또는 None (실패 시)
    """
    md_path = Path(markdown_path)

    if output_path is None:
        output_path = str(md_path.with_suffix(".pdf"))

    # 환경 변수 설정 (CHROME_PATH가 설정되어 있지 않으면 자동 탐지)
    env = os.environ.copy()
    if "CHROME_PATH" not in env:
        chrome_path = _find_chrome_path()
        if chrome_path:
            env["CHROME_PATH"] = chrome_path

    try:
        result = subprocess.run(
            ["marp", str(md_path), "--pdf", "-o", output_path, "--allow-local-files"],
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode == 0:
            return output_path
        else:
            print(f"Marp PDF 변환 실패: {result.stderr}")
            return None

    except FileNotFoundError:
        print("marp-cli가 설치되어 있지 않습니다. 'npm install -g @marp-team/marp-cli'로 설치하세요.")
        return None


def check_marp_installed() -> bool:
    """Marp CLI 설치 여부 확인"""
    return shutil.which("marp") is not None


def export_to_pdf(
    slide_contents: List[Dict[str, Any]],
    output_path: str,
    title: str = "프레젠테이션",
    theme: str = "default",
    keep_markdown: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    슬라이드 콘텐츠를 바로 PDF로 변환 (원스텝)

    Args:
        slide_contents: 슬라이드 콘텐츠 리스트
        output_path: 출력 PDF 파일 경로
        title: 프레젠테이션 제목
        theme: Marp 테마
        keep_markdown: Markdown 파일도 함께 저장할지 여부

    Returns:
        (PDF 경로, Markdown 경로) 튜플. 실패 시 None
    """
    if not check_marp_installed():
        print("marp-cli가 설치되어 있지 않습니다. 'npm install -g @marp-team/marp-cli'로 설치하세요.")
        return None, None

    pdf_path = Path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if keep_markdown:
        md_path = pdf_path.with_suffix(".md")
        export_to_file(slide_contents, str(md_path), title, theme)
        result = convert_to_pdf(str(md_path), str(pdf_path))
        return result, str(md_path) if result else (None, None)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            markdown = to_marp_markdown(slide_contents, title, theme)
            f.write(markdown)
            temp_md = f.name

        try:
            result = convert_to_pdf(temp_md, str(pdf_path))
            return result, None
        finally:
            Path(temp_md).unlink(missing_ok=True)


def _find_slide_contents(data: Dict[str, Any]) -> Tuple[List[Dict], str]:
    """
    에이전트 출력에서 slide_contents를 찾아 반환

    Returns:
        (slide_contents 리스트, 제목)
    """
    # 1. 최상위에 있는 경우
    if "slide_contents" in data and data["slide_contents"]:
        return data["slide_contents"], data.get("title", "프레젠테이션")

    # 2. development.materials[i].slide_contents에 있는 경우 (ADDIE 구조)
    development = data.get("development", {})
    materials = development.get("materials", [])

    all_slides = []
    title = "프레젠테이션"

    if isinstance(materials, list):
        for mat in materials:
            if isinstance(mat, dict):
                slides = mat.get("slide_contents", [])
                if slides:
                    all_slides.extend(slides)
                    if not title or title == "프레젠테이션":
                        title = mat.get("title", title)

    return all_slides, title


def export_from_json(
    json_path: str,
    output_dir: Optional[str] = None,
    formats: List[str] = ["md", "pdf"]
) -> Dict[str, Optional[str]]:
    """
    에이전트 출력 JSON에서 슬라이드를 추출하여 내보내기

    Args:
        json_path: 에이전트 출력 JSON 파일 경로
        output_dir: 출력 디렉토리 (없으면 JSON 파일과 같은 위치)
        formats: 출력 형식 리스트 ["md", "pdf"]

    Returns:
        {"md": md_path, "pdf": pdf_path} 형식의 딕셔너리
    """
    json_file = Path(json_path)

    if not json_file.exists():
        print(f"파일을 찾을 수 없습니다: {json_path}")
        return {"md": None, "pdf": None}

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    slide_contents, title = _find_slide_contents(data)

    if not slide_contents:
        print(f"슬라이드 콘텐츠가 없습니다: {json_path}")
        return {"md": None, "pdf": None}

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = json_file.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = json_file.stem.replace("_output", "_slides")

    result = {"md": None, "pdf": None}

    if "md" in formats:
        md_path = out_dir / f"{base_name}.md"
        result["md"] = export_to_file(slide_contents, str(md_path), title)
        print(f"Markdown 생성: {result['md']}")

    if "pdf" in formats:
        if not check_marp_installed():
            print("PDF 생성 건너뜀 (marp-cli 미설치)")
        else:
            pdf_path = out_dir / f"{base_name}.pdf"
            if result["md"]:
                result["pdf"] = convert_to_pdf(result["md"], str(pdf_path))
            else:
                pdf_result, _ = export_to_pdf(slide_contents, str(pdf_path), title)
                result["pdf"] = pdf_result

            if result["pdf"]:
                print(f"PDF 생성: {result['pdf']}")

    return result


def batch_export_slides(
    results_dir: str,
    output_dir: Optional[str] = None,
    formats: List[str] = ["md", "pdf"]
) -> List[Dict[str, Any]]:
    """
    벤치마크 결과 디렉토리에서 모든 슬라이드를 일괄 내보내기

    Args:
        results_dir: 벤치마크 결과 디렉토리 (예: results/benchmark_20251217_005042)
        output_dir: 출력 디렉토리 (없으면 각 에이전트 결과 폴더에 저장)
        formats: 출력 형식 리스트

    Returns:
        처리 결과 리스트
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"결과 디렉토리를 찾을 수 없습니다: {results_dir}")
        return []

    results = []

    for output_file in results_path.rglob("*_output.json"):
        print(f"\n처리 중: {output_file.relative_to(results_path)}")

        if output_dir:
            relative = output_file.parent.relative_to(results_path)
            target_dir = Path(output_dir) / relative
        else:
            target_dir = output_file.parent

        export_result = export_from_json(
            str(output_file),
            str(target_dir),
            formats
        )

        results.append({
            "source": str(output_file),
            "md": export_result.get("md"),
            "pdf": export_result.get("pdf")
        })

    print(f"\n총 {len(results)}개 파일 처리 완료")
    return results
