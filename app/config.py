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
    TOP_K: int = 5


settings = Settings()
