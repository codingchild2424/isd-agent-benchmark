"""슬라이드 콘텐츠 모델"""
from typing import List, Optional
from pydantic import BaseModel, Field


class SlideContent(BaseModel):
    """개별 슬라이드 콘텐츠"""
    slide_number: int = Field(..., description="슬라이드 번호")
    title: str = Field(..., description="슬라이드 제목")
    bullet_points: List[str] = Field(default_factory=list, description="핵심 내용 (3-5개)")
    speaker_notes: Optional[str] = Field(default=None, description="발표자 노트")
    visual_suggestion: Optional[str] = Field(default=None, description="권장 시각 자료")


class Material(BaseModel):
    """학습 자료"""
    type: str = Field(..., description="자료 유형 (PPT, 동영상, 퀴즈 등)")
    title: str = Field(..., description="자료 제목")
    description: Optional[str] = Field(default=None, description="자료 설명")
    slides: Optional[int] = Field(default=None, description="슬라이드 수 (PPT인 경우)")
    duration: Optional[str] = Field(default=None, description="재생 시간 (동영상인 경우)")
    questions: Optional[int] = Field(default=None, description="문항 수 (퀴즈인 경우)")
    pages: Optional[int] = Field(default=None, description="페이지 수")
    slide_contents: Optional[List[SlideContent]] = Field(
        default=None,
        description="슬라이드별 상세 콘텐츠 (PPT인 경우)"
    )
