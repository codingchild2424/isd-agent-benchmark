"""
IDLD 데이터셋 로더 및 시나리오 생성 지원

Instructional Design Literature Database (IDLD)에서
교수설계 관련 논문 데이터를 로드하고, 일반화된 시나리오 생성을 지원합니다.

주의: 논문 식별 정보(DOI, 저자 등)는 시나리오에 포함되지 않으며,
별도 매핑 파일로 관리됩니다.
"""

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Iterator
from datetime import datetime


@dataclass
class IDLDRecord:
    """IDLD 개별 레코드"""
    no: int                                  # 원본 번호
    year: int                                # 출판 연도
    title: str                               # 논문 제목
    abstract: str                            # 초록
    keywords: Set[str] = field(default_factory=set)  # 저자 키워드

    # 메타데이터 (시나리오에 포함하지 않음)
    authors: str = ""
    doi: str = ""
    source_title: str = ""
    link: str = ""
    document_type: str = ""

    def to_source_mapping(self) -> Dict:
        """출처 매핑용 딕셔너리 반환 (시나리오와 분리 보관)"""
        return {
            "source_no": self.no,
            "year": self.year,
            "original_title": self.title,
            "authors": self.authors,
            "doi": self.doi,
            "source_title": self.source_title,
            "link": self.link,
            "document_type": self.document_type,
            "keywords": list(self.keywords),
        }

    def get_context_hints(self) -> Dict:
        """
        시나리오 생성을 위한 컨텍스트 힌트 추출
        (논문 식별 정보 제외, 교육적 맥락만 포함)
        """
        return {
            "abstract": self.abstract,
            "keywords": list(self.keywords),
            "year": self.year,  # 최신 트렌드 반영용
        }


@dataclass
class ScenarioSchema:
    """
    시나리오 JSON 스키마 (기존 형식 준수)

    variant_type 필드 (Issue #43):
        - "idld_aligned": IDLD 논문의 맥락과 일치하는 전형적인 조합
        - "context_variant": 컨텍스트를 의도적으로 변형한 확장 케이스
    """
    scenario_id: str
    title: str
    context: Dict[str, any]
    learning_goals: List[str]
    constraints: Dict[str, any]
    difficulty: str
    domain: str
    variant_type: str = "idld_aligned"  # "idld_aligned" | "context_variant"

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "variant_type": self.variant_type,
            "title": self.title,
            "context": self.context,
            "learning_goals": self.learning_goals,
            "constraints": self.constraints,
            "difficulty": self.difficulty,
            "domain": self.domain,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def create_empty(cls, scenario_id: str, variant_type: str = "realistic") -> "ScenarioSchema":
        """빈 시나리오 템플릿 생성"""
        return cls(
            scenario_id=scenario_id,
            title="",
            context={
                "target_audience": "",
                "prior_knowledge": "",
                "duration": "",
                "learning_environment": "",
                "class_size": None,
                "additional_context": "",
            },
            learning_goals=[],
            constraints={
                "budget": None,
                "resources": [],
                "accessibility": None,
                "language": "ko",
            },
            difficulty="medium",
            domain="",
            variant_type=variant_type,
        )


@dataclass
class SourceMapping:
    """출처 매핑 관리"""
    mappings: Dict[str, Dict] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add(self, scenario_id: str, record: IDLDRecord) -> None:
        """시나리오-출처 매핑 추가"""
        self.mappings[scenario_id] = record.to_source_mapping()

    def get_source(self, scenario_id: str) -> Optional[Dict]:
        """시나리오의 원본 출처 조회"""
        return self.mappings.get(scenario_id)

    def to_dict(self) -> Dict:
        return {
            "created_at": self.created_at,
            "total_count": len(self.mappings),
            "mappings": self.mappings,
        }

    def save(self, path: str) -> None:
        """매핑 파일 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SourceMapping":
        """매핑 파일 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mapping = cls(created_at=data.get("created_at", ""))
        mapping.mappings = data.get("mappings", {})
        return mapping


class IDLDDataset:
    """IDLD 데이터셋 로더"""

    def __init__(self, csv_path: Optional[str] = None):
        """
        Args:
            csv_path: IDLD CSV 파일 경로. None이면 기본 경로 사용.
        """
        self.records: List[IDLDRecord] = []
        self.keyword_index: Dict[str, List[int]] = {}  # 키워드 → 레코드 인덱스

        if csv_path:
            self.load_from_csv(csv_path)
        else:
            default_path = self._get_default_csv_path()
            if default_path.exists():
                self.load_from_csv(str(default_path))

    @staticmethod
    def _get_default_csv_path() -> Path:
        """기본 CSV 경로 반환"""
        current_dir = Path(__file__).parent
        benchmark_dir = current_dir.parent.parent
        return benchmark_dir / "scenarios" / "IDLD.xlsx - sheet1.csv"

    def load_from_csv(self, path: str) -> "IDLDDataset":
        """CSV 파일에서 데이터 로드"""
        self.records = []
        self.keyword_index = {}

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # 키워드 파싱 (세미콜론 분리 → 정규화)
                keywords_raw = row.get('Author Keywords', '')
                keywords = self._parse_keywords(keywords_raw)

                # 연도 파싱
                try:
                    year = int(row.get('Year', 0))
                except ValueError:
                    year = 0

                record = IDLDRecord(
                    no=int(row.get('No', idx + 1)),
                    year=year,
                    title=row.get('Title', ''),
                    abstract=row.get('Abstract', ''),
                    keywords=keywords,
                    authors=row.get('Authors', ''),
                    doi=row.get('DOI', ''),
                    source_title=row.get('Source title', ''),
                    link=row.get('Link', ''),
                    document_type=row.get('Document Type', ''),
                )
                self.records.append(record)

                # 키워드 인덱스 구축
                for kw in keywords:
                    if kw not in self.keyword_index:
                        self.keyword_index[kw] = []
                    self.keyword_index[kw].append(len(self.records) - 1)

        return self

    @staticmethod
    def _parse_keywords(keywords_str: str) -> Set[str]:
        """키워드 문자열 파싱 및 정규화"""
        if not keywords_str or not keywords_str.strip():
            return set()

        keywords = set()
        for kw in keywords_str.split(';'):
            normalized = kw.strip().lower()
            if normalized:
                keywords.add(normalized)

        return keywords

    # =========================================================================
    # 데이터 조회
    # =========================================================================

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> IDLDRecord:
        return self.records[idx]

    def __iter__(self) -> Iterator[IDLDRecord]:
        return iter(self.records)

    def get_by_no(self, no: int) -> Optional[IDLDRecord]:
        """원본 번호로 레코드 조회"""
        for record in self.records:
            if record.no == no:
                return record
        return None

    def get_all_keywords(self) -> Set[str]:
        """전체 키워드 집합 반환"""
        return set(self.keyword_index.keys())

    def get_keyword_frequency(self) -> Dict[str, int]:
        """키워드별 빈도 반환"""
        return {kw: len(indices) for kw, indices in self.keyword_index.items()}

    # =========================================================================
    # 필터링 및 샘플링
    # =========================================================================

    def filter_by_keyword(self, keyword: str) -> List[IDLDRecord]:
        """특정 키워드를 포함하는 레코드 필터링"""
        keyword_lower = keyword.lower()
        indices = self.keyword_index.get(keyword_lower, [])
        return [self.records[i] for i in indices]

    def filter_by_keywords(
        self,
        keywords: List[str],
        match_all: bool = False
    ) -> List[IDLDRecord]:
        """
        여러 키워드로 필터링

        Args:
            keywords: 검색할 키워드 리스트
            match_all: True면 모든 키워드 포함, False면 하나라도 포함
        """
        keywords_lower = {kw.lower() for kw in keywords}

        results = []
        for record in self.records:
            record_kws = record.keywords
            if match_all:
                if keywords_lower.issubset(record_kws):
                    results.append(record)
            else:
                if keywords_lower & record_kws:
                    results.append(record)

        return results

    def filter_by_year(
        self,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> List[IDLDRecord]:
        """연도 범위로 필터링"""
        results = []
        for record in self.records:
            if min_year and record.year < min_year:
                continue
            if max_year and record.year > max_year:
                continue
            results.append(record)
        return results

    def filter_has_abstract(self, min_length: int = 100) -> List[IDLDRecord]:
        """충분한 길이의 초록이 있는 레코드만 필터링"""
        return [r for r in self.records if len(r.abstract) >= min_length]

    def sample(self, n: int, has_abstract: bool = True) -> List[IDLDRecord]:
        """무작위 샘플링"""
        pool = self.filter_has_abstract() if has_abstract else self.records
        return random.sample(pool, min(n, len(pool)))

    def sample_by_keyword(
        self,
        keyword: str,
        n: int,
        has_abstract: bool = True
    ) -> List[IDLDRecord]:
        """특정 키워드를 포함하는 레코드에서 샘플링"""
        filtered = self.filter_by_keyword(keyword)
        if has_abstract:
            filtered = [r for r in filtered if len(r.abstract) >= 100]
        return random.sample(filtered, min(n, len(filtered)))

    # =========================================================================
    # 시나리오 생성 지원
    # =========================================================================

    def prepare_for_scenario_generation(
        self,
        n: int,
        min_abstract_length: int = 200,
        recent_years_only: bool = False
    ) -> List[Dict]:
        """
        시나리오 생성을 위한 데이터 준비

        Args:
            n: 생성할 시나리오 수
            min_abstract_length: 최소 초록 길이
            recent_years_only: True면 최근 5년 논문만

        Returns:
            시나리오 생성용 힌트 리스트 (논문 식별 정보 제외)
        """
        # 필터링
        pool = [r for r in self.records if len(r.abstract) >= min_abstract_length]

        if recent_years_only:
            current_year = datetime.now().year
            pool = [r for r in pool if r.year >= current_year - 5]

        # 샘플링
        sampled = random.sample(pool, min(n, len(pool)))

        # 힌트 추출 (식별 정보 제외)
        return [
            {
                "record_no": r.no,  # 매핑용 (시나리오에는 포함 안 함)
                "hints": r.get_context_hints(),
            }
            for r in sampled
        ]

    # =========================================================================
    # 통계 정보
    # =========================================================================

    def summary(self) -> Dict:
        """데이터셋 요약 정보"""
        years = [r.year for r in self.records if r.year > 0]
        abstracts = [r.abstract for r in self.records]
        abstract_lengths = [len(a) for a in abstracts if a]

        keywords_with_data = [r for r in self.records if r.keywords]

        return {
            "total_records": len(self.records),
            "unique_keywords": len(self.keyword_index),
            "records_with_keywords": len(keywords_with_data),
            "records_with_abstract": len([a for a in abstracts if len(a) >= 100]),
            "year_range": {
                "min": min(years) if years else None,
                "max": max(years) if years else None,
            },
            "abstract_length": {
                "min": min(abstract_lengths) if abstract_lengths else 0,
                "max": max(abstract_lengths) if abstract_lengths else 0,
                "avg": sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0,
            },
            "top_keywords": sorted(
                self.get_keyword_frequency().items(),
                key=lambda x: -x[1]
            )[:20],
        }

    def __repr__(self) -> str:
        return f"IDLDDataset(records={len(self.records)}, keywords={len(self.keyword_index)})"
