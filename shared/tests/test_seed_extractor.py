"""시드 추출기 단위 테스트"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from shared.models.seed_extractor import (
    SeedExtractor,
    ScenarioSeed,
    ExtractionStatus,
    EducationLevel,
    SubjectDomain,
    LLMExtractionResult,
    KEYWORD_TO_LEVEL,
    KEYWORD_TO_DOMAIN,
)
from shared.models.idld_dataset import IDLDDataset, IDLDRecord


@pytest.fixture
def csv_path() -> str:
    """테스트용 CSV 경로"""
    current_dir = Path(__file__).parent
    return str(current_dir.parent.parent / "scenarios" / "IDLD.xlsx - sheet1.csv")


@pytest.fixture
def idld_dataset(csv_path: str) -> IDLDDataset:
    """IDLDDataset 인스턴스"""
    return IDLDDataset(csv_path)


@pytest.fixture
def sample_record() -> IDLDRecord:
    """테스트용 샘플 레코드"""
    return IDLDRecord(
        no=1,
        year=2023,
        title="Learning metabolism with virtual reality",
        abstract="""This study examines the effectiveness of virtual reality (VR)
        technology in teaching biochemical metabolism to undergraduate students.
        Using problem-based learning (PBL) approach, students engaged with
        interactive 3D models of metabolic pathways. Results showed significant
        improvement in understanding complex concepts compared to traditional
        lecture-based methods. The study suggests VR-based learning environments
        can enhance student engagement and conceptual understanding in STEM education.""",
        keywords={"virtual reality", "biochemistry", "higher education", "stem", "problem-based learning"},
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM 응답"""
    return LLMExtractionResult(
        topic="생화학 대사 작용",
        topic_english="Biochemical Metabolism",
        pedagogical_method="VR 기반 문제중심학습",
        pedagogical_method_english="VR-based Problem-Based Learning",
        confidence=0.85,
        reasoning="초록에서 VR 기술과 PBL 방식을 사용한 생화학 교육이 명확히 언급됨",
    )


class TestScenarioSeed:
    """ScenarioSeed 데이터 구조 테스트"""

    def test_create_seed(self):
        """시드 생성"""
        seed = ScenarioSeed(
            topic="머신러닝 기초",
            pedagogical_method="프로젝트 기반 학습",
            categories=["Higher Education", "STEM"],
            source_record_no=123,
        )

        assert seed.topic == "머신러닝 기초"
        assert seed.pedagogical_method == "프로젝트 기반 학습"
        assert len(seed.categories) == 2
        assert seed.status == ExtractionStatus.SUCCESS
        assert seed.is_valid()

    def test_seed_to_dict(self):
        """딕셔너리 변환"""
        seed = ScenarioSeed(
            topic="테스트 주제",
            pedagogical_method="테스트 교수법",
            categories=["K-12"],
            source_record_no=1,
            confidence_score=0.9,
        )

        data = seed.to_dict()
        assert data["topic"] == "테스트 주제"
        assert data["status"] == "success"
        assert "extracted_at" in data

    def test_seed_from_dict(self):
        """딕셔너리에서 로드"""
        data = {
            "topic": "로드된 주제",
            "pedagogical_method": "로드된 교수법",
            "categories": ["STEM"],
            "source_record_no": 42,
            "status": "needs_review",
            "warnings": ["낮은 신뢰도"],
            "confidence_score": 0.5,
            "extracted_at": "2023-01-01T00:00:00",
        }

        seed = ScenarioSeed.from_dict(data)
        assert seed.topic == "로드된 주제"
        assert seed.status == ExtractionStatus.NEEDS_REVIEW
        assert seed.needs_review()

    def test_seed_validation(self):
        """유효성 검사"""
        # 유효한 시드
        valid_seed = ScenarioSeed(
            topic="유효한 주제",
            pedagogical_method="유효한 교수법",
        )
        assert valid_seed.is_valid()

        # 빈 주제
        invalid_seed = ScenarioSeed(
            topic="",
            pedagogical_method="교수법",
        )
        assert not invalid_seed.is_valid()

        # 실패 상태
        failed_seed = ScenarioSeed(
            topic="주제",
            pedagogical_method="교수법",
            status=ExtractionStatus.FAILED,
        )
        assert not failed_seed.is_valid()


class TestCategoryMapping:
    """카테고리 매핑 테스트"""

    def test_education_level_mapping(self):
        """교육 수준 매핑"""
        assert "higher education" in KEYWORD_TO_LEVEL
        assert KEYWORD_TO_LEVEL["higher education"] == EducationLevel.HIGHER_EDUCATION

        assert "k-12" in KEYWORD_TO_LEVEL
        assert KEYWORD_TO_LEVEL["k-12"] == EducationLevel.K12

    def test_subject_domain_mapping(self):
        """분야 매핑"""
        assert "stem" in KEYWORD_TO_DOMAIN
        assert KEYWORD_TO_DOMAIN["stem"] == SubjectDomain.STEM

        assert "machine learning" in KEYWORD_TO_DOMAIN
        assert KEYWORD_TO_DOMAIN["machine learning"] == SubjectDomain.IT


class TestSeedExtractorCategoryClassification:
    """SeedExtractor 카테고리 분류 테스트 (LLM 불필요)"""

    def test_classify_from_keywords(self, sample_record):
        """키워드 기반 카테고리 분류"""
        extractor = SeedExtractor()
        categories = extractor._classify_categories(
            sample_record.keywords,
            sample_record.abstract,
        )

        # STEM과 Higher Education이 포함되어야 함
        assert "Higher Education" in categories
        assert "STEM" in categories

    def test_classify_from_abstract(self):
        """초록 기반 카테고리 분류"""
        extractor = SeedExtractor()
        categories = extractor._classify_categories(
            keywords=set(),
            abstract="This corporate training program uses machine learning for employee development.",
        )

        assert "Corporate Training" in categories
        assert "IT/Computer Science" in categories


class TestSeedExtractorValidation:
    """SeedExtractor 검증 로직 테스트"""

    def test_create_seed_success(self, mock_llm_response):
        """성공적인 시드 생성"""
        extractor = SeedExtractor()
        seed = extractor._create_seed(
            llm_result=mock_llm_response,
            categories=["Higher Education", "STEM"],
            source_record_no=1,
        )

        assert seed.status == ExtractionStatus.SUCCESS
        assert seed.topic == "생화학 대사 작용"
        assert len(seed.warnings) == 0

    def test_create_seed_with_low_confidence(self):
        """낮은 신뢰도 시드"""
        extractor = SeedExtractor()
        low_confidence_result = LLMExtractionResult(
            topic="모호한 주제",
            topic_english="Vague Topic",
            pedagogical_method="일반적인 방법",
            pedagogical_method_english="General Method",
            confidence=0.3,  # MIN_CONFIDENCE(0.6) 미만
            reasoning="추출 근거 불명확",
        )

        seed = extractor._create_seed(
            llm_result=low_confidence_result,
            categories=[],
            source_record_no=1,
        )

        assert seed.status == ExtractionStatus.NEEDS_REVIEW
        assert any("신뢰도" in w for w in seed.warnings)

    def test_create_seed_with_vague_terms(self):
        """모호한 표현 감지"""
        extractor = SeedExtractor()
        vague_result = LLMExtractionResult(
            topic="다양한 교육 방법",
            topic_english="Various Educational Methods",
            pedagogical_method="일반적인 학습 접근",
            pedagogical_method_english="General Learning Approach",
            confidence=0.8,
            reasoning="모호한 표현 포함",
        )

        seed = extractor._create_seed(
            llm_result=vague_result,
            categories=[],
            source_record_no=1,
        )

        assert seed.status == ExtractionStatus.NEEDS_REVIEW
        assert any("모호한" in w for w in seed.warnings)

    def test_create_seed_with_long_topic(self):
        """너무 긴 주제 감지"""
        extractor = SeedExtractor()
        long_result = LLMExtractionResult(
            topic="이것은 매우 매우 매우 매우 매우 매우 매우 매우 매우 매우 긴 주제입니다 정말로 너무 깁니다",
            topic_english="This is a very long topic",
            pedagogical_method="정상적인 교수법",
            pedagogical_method_english="Normal Method",
            confidence=0.8,
            reasoning="테스트",
        )

        seed = extractor._create_seed(
            llm_result=long_result,
            categories=[],
            source_record_no=1,
        )

        assert seed.status == ExtractionStatus.NEEDS_REVIEW
        assert any("너무 깁니다" in w for w in seed.warnings)

    def test_create_seed_extraction_failed(self):
        """추출 실패 감지"""
        extractor = SeedExtractor()
        failed_result = LLMExtractionResult(
            topic="추출 실패",
            topic_english="Extraction Failed",
            pedagogical_method="추출 실패",
            pedagogical_method_english="Extraction Failed",
            confidence=0.0,
            reasoning="JSON 파싱 오류",
        )

        seed = extractor._create_seed(
            llm_result=failed_result,
            categories=[],
            source_record_no=1,
        )

        assert seed.status == ExtractionStatus.FAILED


class TestSeedExtractorWithMockLLM:
    """Mock LLM을 사용한 SeedExtractor 테스트"""

    def test_extract_with_mock(self, sample_record, mock_llm_response):
        """Mock LLM으로 추출 테스트"""
        extractor = SeedExtractor()

        # _extract_with_llm 메서드를 Mock으로 대체
        with patch.object(extractor, '_extract_with_llm', return_value=mock_llm_response):
            seed = extractor.extract(sample_record)

        assert seed.topic == "생화학 대사 작용"
        assert seed.pedagogical_method == "VR 기반 문제중심학습"
        assert "Higher Education" in seed.categories
        assert seed.source_record_no == sample_record.no

    def test_extract_batch_with_mock(self, idld_dataset, mock_llm_response):
        """배치 추출 테스트"""
        extractor = SeedExtractor()
        records = idld_dataset.sample(n=3)

        with patch.object(extractor, '_extract_with_llm', return_value=mock_llm_response):
            seeds = extractor.extract_batch(records)

        assert len(seeds) == 3
        for seed in seeds:
            assert seed.topic == "생화학 대사 작용"

    def test_extract_batch_skip_on_error(self, sample_record):
        """에러 시 건너뛰기"""
        extractor = SeedExtractor()
        records = [sample_record, sample_record, sample_record]

        # 두 번째 호출에서 에러 발생
        def mock_extract(abstract):
            if hasattr(mock_extract, 'call_count'):
                mock_extract.call_count += 1
            else:
                mock_extract.call_count = 1

            if mock_extract.call_count == 2:
                raise Exception("Test error")

            return LLMExtractionResult(
                topic="정상 주제",
                topic_english="Normal Topic",
                pedagogical_method="정상 교수법",
                pedagogical_method_english="Normal Method",
                confidence=0.9,
                reasoning="테스트",
            )

        with patch.object(extractor, '_extract_with_llm', side_effect=mock_extract):
            seeds = extractor.extract_batch(records, skip_on_error=True)

        assert len(seeds) == 3
        assert seeds[0].status == ExtractionStatus.SUCCESS
        assert seeds[1].status == ExtractionStatus.FAILED
        assert seeds[2].status == ExtractionStatus.SUCCESS


class TestSeedPersistence:
    """시드 저장/로드 테스트"""

    def test_save_and_load_seeds(self):
        """시드 저장 및 로드"""
        seeds = [
            ScenarioSeed(
                topic=f"주제 {i}",
                pedagogical_method=f"교수법 {i}",
                categories=["STEM"],
                source_record_no=i,
                status=ExtractionStatus.SUCCESS if i % 2 == 0 else ExtractionStatus.NEEDS_REVIEW,
                confidence_score=0.9 if i % 2 == 0 else 0.5,
            )
            for i in range(5)
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        # 저장
        SeedExtractor.save_seeds(seeds, temp_path)

        # 저장된 파일 검증
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["total_count"] == 5
        assert data["success_count"] == 3
        assert data["review_count"] == 2

        # 로드
        loaded_seeds = SeedExtractor.load_seeds(temp_path)
        assert len(loaded_seeds) == 5
        assert loaded_seeds[0].topic == "주제 0"
        assert loaded_seeds[1].status == ExtractionStatus.NEEDS_REVIEW

        # 정리
        Path(temp_path).unlink()


class TestSeedStatistics:
    """시드 통계 테스트"""

    def test_get_stats(self):
        """통계 정보"""
        seeds = [
            ScenarioSeed(
                topic="주제1",
                pedagogical_method="교수법1",
                categories=["STEM", "Higher Education"],
                status=ExtractionStatus.SUCCESS,
                confidence_score=0.9,
            ),
            ScenarioSeed(
                topic="주제2",
                pedagogical_method="교수법2",
                categories=["K-12"],
                status=ExtractionStatus.NEEDS_REVIEW,
                confidence_score=0.5,
            ),
            ScenarioSeed(
                topic="",
                pedagogical_method="",
                status=ExtractionStatus.FAILED,
                confidence_score=0.0,
            ),
        ]

        stats = SeedExtractor.get_stats(seeds)

        assert stats["total"] == 3
        assert stats["success"] == 1
        assert stats["needs_review"] == 1
        assert stats["failed"] == 1
        assert abs(stats["avg_confidence"] - 0.467) < 0.01
        assert "STEM" in stats["categories"]
        assert "Higher Education" in stats["categories"]
        assert "K-12" in stats["categories"]


class TestLLMResponseParsing:
    """LLM 응답 파싱 테스트 (Mock 사용)"""

    def test_parse_json_response(self):
        """정상 JSON 응답 파싱"""
        extractor = SeedExtractor()

        mock_response = Mock()
        mock_response.content = json.dumps({
            "topic": "테스트 주제",
            "topic_english": "Test Topic",
            "pedagogical_method": "테스트 교수법",
            "pedagogical_method_english": "Test Method",
            "confidence": 0.85,
            "reasoning": "테스트 근거",
        })

        # _llm 속성에 직접 Mock 주입
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        result = extractor._extract_with_llm("테스트 초록")

        assert result.topic == "테스트 주제"
        assert result.confidence == 0.85

    def test_parse_json_with_markdown(self):
        """마크다운 코드 블록 내 JSON 파싱"""
        extractor = SeedExtractor()

        mock_response = Mock()
        mock_response.content = """```json
{
    "topic": "마크다운 주제",
    "topic_english": "Markdown Topic",
    "pedagogical_method": "마크다운 교수법",
    "pedagogical_method_english": "Markdown Method",
    "confidence": 0.75,
    "reasoning": "마크다운 테스트"
}
```"""

        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        result = extractor._extract_with_llm("테스트 초록")

        assert result.topic == "마크다운 주제"

    def test_parse_invalid_json(self):
        """잘못된 JSON 응답 처리"""
        extractor = SeedExtractor()

        mock_response = Mock()
        mock_response.content = "This is not valid JSON"

        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        result = extractor._extract_with_llm("테스트 초록")

        # 파싱 실패 시 기본값 반환
        assert result.topic == "추출 실패"
        assert result.confidence == 0.0


@pytest.mark.integration
class TestIntegrationWithRealLLM:
    """실제 LLM 연동 테스트 (API 키 필요)"""

    @pytest.mark.skip(reason="실제 LLM API 호출 필요 - 수동 테스트용")
    def test_extract_real(self, sample_record):
        """실제 LLM으로 추출"""
        extractor = SeedExtractor()
        seed = extractor.extract(sample_record)

        assert seed.topic
        assert seed.pedagogical_method
        assert seed.confidence_score > 0
        print(f"\n추출 결과:")
        print(f"  주제: {seed.topic}")
        print(f"  교수법: {seed.pedagogical_method}")
        print(f"  카테고리: {seed.categories}")
        print(f"  신뢰도: {seed.confidence_score}")
        print(f"  상태: {seed.status.value}")
