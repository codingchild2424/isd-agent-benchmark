"""IDLD 데이터셋 및 시나리오 생성기 단위 테스트"""

import pytest
import json
import tempfile
from pathlib import Path
from shared.models.idld_dataset import (
    IDLDDataset,
    IDLDRecord,
    ScenarioSchema,
    SourceMapping,
)
from shared.models.scenario_generator import (
    ScenarioGenerator,
    ScenarioGenerationRequest,
)
from shared.models.context_matrix import ContextMatrix, ContextCombination


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
def context_matrix() -> ContextMatrix:
    """ContextMatrix 인스턴스"""
    return ContextMatrix()


@pytest.fixture
def scenario_generator(idld_dataset, context_matrix) -> ScenarioGenerator:
    """ScenarioGenerator 인스턴스"""
    return ScenarioGenerator(idld_dataset, context_matrix)


class TestIDLDDatasetLoad:
    """IDLD 데이터셋 로드 테스트"""

    def test_load_csv(self, idld_dataset: IDLDDataset):
        """CSV 파일 로드 검증"""
        assert len(idld_dataset) > 10000
        assert len(idld_dataset.keyword_index) > 0

    def test_record_structure(self, idld_dataset: IDLDDataset):
        """레코드 구조 검증"""
        record = idld_dataset[0]
        assert isinstance(record, IDLDRecord)
        assert record.no > 0
        assert record.year > 2000
        assert len(record.title) > 0
        assert len(record.abstract) > 0

    def test_keyword_parsing(self, idld_dataset: IDLDDataset):
        """키워드 파싱 검증"""
        # 키워드가 있는 레코드 찾기
        records_with_keywords = [r for r in idld_dataset if r.keywords]
        assert len(records_with_keywords) > 0

        record = records_with_keywords[0]
        # 키워드가 소문자로 정규화되었는지 확인
        for kw in record.keywords:
            assert kw == kw.lower()

    def test_keyword_index(self, idld_dataset: IDLDDataset):
        """키워드 인덱스 구축 확인"""
        all_keywords = idld_dataset.get_all_keywords()
        assert len(all_keywords) > 100

        # 빈도 확인
        freq = idld_dataset.get_keyword_frequency()
        assert "instructional design" in freq or len(freq) > 0


class TestIDLDFiltering:
    """IDLD 필터링 테스트"""

    def test_filter_by_keyword(self, idld_dataset: IDLDDataset):
        """키워드 필터링"""
        # 일반적인 키워드로 테스트
        results = idld_dataset.filter_by_keyword("virtual reality")
        # 결과가 없어도 에러 없이 빈 리스트 반환
        assert isinstance(results, list)

    def test_filter_has_abstract(self, idld_dataset: IDLDDataset):
        """초록 길이 필터링"""
        results = idld_dataset.filter_has_abstract(min_length=200)
        assert len(results) > 0
        for r in results:
            assert len(r.abstract) >= 200

    def test_sample(self, idld_dataset: IDLDDataset):
        """무작위 샘플링"""
        samples = idld_dataset.sample(n=10)
        assert len(samples) == 10
        # 모든 샘플이 유효한 초록을 가짐
        for s in samples:
            assert len(s.abstract) >= 100


class TestIDLDRecordMethods:
    """IDLDRecord 메서드 테스트"""

    def test_to_source_mapping(self, idld_dataset: IDLDDataset):
        """출처 매핑 변환"""
        record = idld_dataset[0]
        mapping = record.to_source_mapping()

        assert "source_no" in mapping
        assert "original_title" in mapping
        assert "doi" in mapping
        assert "keywords" in mapping
        assert isinstance(mapping["keywords"], list)

    def test_get_context_hints(self, idld_dataset: IDLDDataset):
        """컨텍스트 힌트 추출 (식별정보 제외)"""
        record = idld_dataset[0]
        hints = record.get_context_hints()

        # 교육적 맥락만 포함
        assert "abstract" in hints
        assert "keywords" in hints

        # 식별 정보 제외 확인
        assert "doi" not in hints
        assert "authors" not in hints
        assert "original_title" not in hints


class TestScenarioSchema:
    """시나리오 스키마 테스트"""

    def test_create_empty(self):
        """빈 시나리오 생성"""
        scenario = ScenarioSchema.create_empty("TEST-001")
        assert scenario.scenario_id == "TEST-001"
        assert scenario.title == ""
        assert "target_audience" in scenario.context

    def test_to_json(self):
        """JSON 변환"""
        scenario = ScenarioSchema(
            scenario_id="TEST-001",
            title="테스트 교육",
            context={
                "target_audience": "대학생",
                "prior_knowledge": "기초 지식",
                "duration": "2시간",
                "learning_environment": "대면",
                "class_size": 20,
                "additional_context": "",
            },
            learning_goals=["목표1", "목표2"],
            constraints={
                "budget": "medium",
                "resources": ["PPT"],
                "accessibility": None,
                "language": "ko",
            },
            difficulty="easy",
            domain="교육",
        )

        json_str = scenario.to_json()
        parsed = json.loads(json_str)
        assert parsed["scenario_id"] == "TEST-001"
        assert parsed["title"] == "테스트 교육"


class TestSourceMapping:
    """출처 매핑 테스트"""

    def test_add_and_get(self, idld_dataset: IDLDDataset):
        """매핑 추가 및 조회"""
        mapping = SourceMapping()
        record = idld_dataset[0]

        mapping.add("IDLD-0001", record)

        source = mapping.get_source("IDLD-0001")
        assert source is not None
        assert source["source_no"] == record.no

    def test_save_and_load(self, idld_dataset: IDLDDataset):
        """매핑 저장 및 로드"""
        mapping = SourceMapping()
        record = idld_dataset[0]
        mapping.add("IDLD-0001", record)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        mapping.save(temp_path)

        loaded = SourceMapping.load(temp_path)
        assert "IDLD-0001" in loaded.mappings
        assert loaded.mappings["IDLD-0001"]["source_no"] == record.no

        # 정리
        Path(temp_path).unlink()


class TestScenarioGenerator:
    """시나리오 생성기 테스트"""

    def test_prepare_generation_requests(self, scenario_generator: ScenarioGenerator):
        """생성 요청 준비"""
        requests = scenario_generator.prepare_generation_requests(n=10)

        assert len(requests) == 10
        for req in requests:
            assert isinstance(req, ScenarioGenerationRequest)
            assert isinstance(req.idld_record, IDLDRecord)
            assert isinstance(req.context_combination, ContextCombination)
            assert req.difficulty in ["easy", "medium", "hard"]

    def test_llm_prompt_context(self, scenario_generator: ScenarioGenerator):
        """LLM 프롬프트 컨텍스트 (식별정보 제외 확인)"""
        requests = scenario_generator.prepare_generation_requests(n=1)
        req = requests[0]

        context = req.to_llm_prompt_context()

        # 교육적 맥락 포함
        assert "source_abstract" in context
        assert "source_keywords" in context
        assert "context" in context

        # 식별 정보 제외
        assert "doi" not in context
        assert "authors" not in context
        assert "original_title" not in context

    def test_generate_scenario_id(self, scenario_generator: ScenarioGenerator):
        """시나리오 ID 생성"""
        id1 = scenario_generator.generate_scenario_id("IDLD", 1)
        id2 = scenario_generator.generate_scenario_id("IDLD", 999)

        assert id1 == "IDLD-0001"
        assert id2 == "IDLD-0999"

    def test_register_source_mapping(self, scenario_generator: ScenarioGenerator):
        """출처 매핑 등록"""
        record = scenario_generator.idld[0]
        scenario_generator.register_source_mapping("IDLD-0001", record)

        source = scenario_generator.get_source_for_scenario("IDLD-0001")
        assert source is not None
        assert source["source_no"] == record.no

    def test_ablation_requests(self, scenario_generator: ScenarioGenerator):
        """Ablation 요청 준비"""
        record = scenario_generator.idld.sample(n=1)[0]
        base_context = ContextCombination(
            learner_age="20대",
            institution_type="대학교(학부)",
        )

        requests = scenario_generator.prepare_ablation_requests(
            record,
            base_context,
            vary_dimension="학습자 특성"
        )

        assert len(requests) > 0
        # 모든 요청이 동일한 IDLD 레코드 사용
        for req in requests:
            assert req.idld_record.no == record.no


class TestIntegration:
    """통합 테스트"""

    def test_full_workflow(self, scenario_generator: ScenarioGenerator):
        """전체 워크플로우 테스트"""
        # 1. 생성 요청 준비
        requests = scenario_generator.prepare_generation_requests(n=5)
        assert len(requests) == 5

        # 2. 각 요청에 대해 시나리오 ID 생성 및 매핑 등록
        for i, req in enumerate(requests):
            scenario_id = scenario_generator.generate_scenario_id("IDLD", i + 1)
            scenario_generator.register_source_mapping(scenario_id, req.idld_record)

        # 3. 매핑 확인
        assert len(scenario_generator.source_mapping.mappings) == 5

        # 4. 요약 확인
        summary = scenario_generator.summary()
        assert summary["registered_mappings"] == 5

    def test_prompt_template(self, scenario_generator: ScenarioGenerator):
        """프롬프트 템플릿 확인"""
        template = scenario_generator.get_generation_prompt_template()

        # 필수 요소 포함 확인
        assert "{source_abstract}" in template
        assert "{difficulty}" in template
        assert "JSON" in template
        assert "DOI" in template  # DOI 포함하지 말라는 지시
