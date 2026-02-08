"""에이전트 기본 클래스"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel


# API 설정
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1/solar"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
UPSTAGE_DEFAULT_MODEL = "solar-mini"


class AgentConfig(BaseModel):
    """에이전트 설정"""
    model: str = UPSTAGE_DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 4096
    provider: str = "upstage"  # "upstage", "openai", "anthropic", or "openrouter"


class BaseAgent(ABC):
    """에이전트 기본 클래스"""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._llm = None

    @property
    def llm(self):
        """LLM 인스턴스 반환 (지연 초기화)"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self):
        """LLM 인스턴스 생성"""
        if self.config.provider == "anthropic":
            return ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        elif self.config.provider == "openrouter":
            # OpenRouter API (OpenAI 호환)
            return ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=OPENROUTER_BASE_URL,
            )
        elif self.config.provider == "upstage":
            # Upstage API (OpenAI 호환)
            return ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=os.getenv("UPSTAGE_API_KEY"),
                base_url=UPSTAGE_BASE_URL,
            )
        else:
            # OpenAI
            return ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """에이전트 실행"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """에이전트 이름"""
        pass

    @property
    @abstractmethod
    def role(self) -> str:
        """에이전트 역할 설명"""
        pass
