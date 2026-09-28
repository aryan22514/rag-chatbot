from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GEMINI_API_KEY: str
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    CHROMA_DIR: str = "./data/chroma"
    COLLECTION_NAME: str = "documents"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    EMBEDDING_DIM: int = 768
    LLM_MODEL: str = "gemini-3.8-flash"
    # Tried in order when the main model hits its (free-tier) rate limit
    LLM_FALLBACK_MODELS: str = "gemini-3.6-flash,gemini-3.5-flash-lite"
    # Suggested questions don't need the strongest model; keep them off its quota
    SUGGESTION_MODEL: str = "gemini-3.5-flash-lite"
    TOP_K: int = 5
    SUGGESTION_COUNT: int = 4

    @property
    def llm_models(self) -> list[str]:
        """Main model first, then fallbacks — without duplicates."""
        extra = [m.strip() for m in self.LLM_FALLBACK_MODELS.split(",") if m.strip()]
        return list(dict.fromkeys([self.LLM_MODEL, *extra]))

    @property
    def suggestion_models(self) -> list[str]:
        """Cheapest first, so suggestions rarely touch the main model's quota."""
        return list(dict.fromkeys([self.SUGGESTION_MODEL, *reversed(self.llm_models)]))


settings = Settings()
