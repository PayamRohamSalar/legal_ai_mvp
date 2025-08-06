"""
LLM configuration module.
Supports both API-based and local model configurations using Pydantic.
"""

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings
from typing import Optional, Literal, Dict, Any
from enum import Enum

class LLMProvider(str, Enum):
    """Enumeration for supported LLM providers."""
    GOOGLE_GEMINI = "google_gemini"
    OPENAI = "openai"
    LOCAL = "local" # A generic local provider

class BaseLLMConfig(BaseModel):
    """Base configuration shared by all LLM providers."""
    provider: LLMProvider
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Controls randomness. Lower is more deterministic.")
    max_tokens: int = Field(default=2048, ge=1, description="Maximum number of tokens to generate.")
    timeout: int = Field(default=60, description="Request timeout in seconds.")

class GoogleGeminiConfig(BaseLLMConfig):
    """Configuration specific to Google Gemini API."""
    provider: Literal[LLMProvider.GOOGLE_GEMINI] = LLMProvider.GOOGLE_GEMINI
    model_name: str = Field(default="gemini-1.5-flash-latest", description="Specific Gemini model to use.")
    api_key: SecretStr = Field(..., description="Google API Key.")

class OpenAIConfig(BaseLLMConfig):
    """Configuration specific to OpenAI API."""
    provider: Literal[LLMProvider.OPENAI] = LLMProvider.OPENAI
    model_name: str = Field(default="gpt-4o", description="Specific OpenAI model to use.")
    api_key: SecretStr = Field(..., description="OpenAI API Key.")

class LocalLLMConfig(BaseLLMConfig):
    """Configuration for a local model (e.g., served via Ollama or a local server)."""
    provider: Literal[LLMProvider.LOCAL] = LLMProvider.LOCAL
    model_name: str = Field(default="llama3", description="Name of the local model.")
    api_base: str = Field(default="http://localhost:11434/v1", description="API endpoint for the local LLM server.")

class LLMSettings(BaseSettings):
    """
    Main LLM settings container.
    Loads settings from environment variables.
    """
    # The active LLM provider to be used by the application
    llm_provider: LLMProvider = Field(default=LLMProvider.GOOGLE_GEMINI, env="LLM_PROVIDER")

    # API Keys
    google_api_key: Optional[SecretStr] = Field(None, env="GOOGLE_API_KEY")
    openai_api_key: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    
    # Prompt Template
    # This is the core instruction for the AI assistant.
    system_prompt_template: str = """
You are a 'Specialized Legal Text Retrieval Assistant' for Iran's research and technology domain.
Your persona is a tool for accessing information, NOT a lawyer or a conversational chatbot.

CORE INSTRUCTIONS:
1.  **Strictly Adhere to Source**: Base your answers ONLY on the provided legal text (Context). DO NOT use any external knowledge. If the answer is not in the context, state that you cannot answer based on the provided documents.
2.  **Verbatim Quotation**: Prioritize quoting the relevant legal text verbatim. Summarize only when necessary and ensure the summary does not alter the legal meaning.
3.  **Precise Citations**: You MUST include a precise citation (e.g.,) immediately after the information it supports.
4.  **Neutral and Formal Tone**: Maintain a formal, objective, and neutral tone. Do not offer opinions, interpretations, or legal advice.
5.  **Standard Format**: Respond in the format: Citation → Body Text → Legal Disclaimer.

CONTEXT FROM LEGAL DOCUMENTS:
---
{context}
---

USER'S QUESTION: {question}

ASSISTANT'S RESPONSE (in Persian):
"""

    def get_active_config(self) -> BaseLLMConfig:
        """
        Dynamically gets the configuration for the active LLM provider.
        Raises ValueError if the configuration is incomplete.
        """
        if self.llm_provider == LLMProvider.GOOGLE_GEMINI:
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not set for the active provider.")
            return GoogleGeminiConfig(api_key=self.google_api_key)

        if self.llm_provider == LLMProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set for the active provider.")
            return OpenAIConfig(api_key=self.openai_api_key)
            
        if self.llm_provider == LLMProvider.LOCAL:
            return LocalLLMConfig()

        raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Using SecretStr for API keys helps prevent accidental logging
        # We need to tell Pydantic to encode these as strings when serializing
        json_encoders = {SecretStr: lambda v: v.get_secret_value() if v else None}


# Global instance to be imported by other modules
llm_settings = LLMSettings()