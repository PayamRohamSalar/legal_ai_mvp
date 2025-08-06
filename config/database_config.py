"""
Database configuration module.
Handles PostgreSQL, ChromaDB, and Redis configurations using Pydantic.
This module consolidates all data-layer configurations.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
from pathlib import Path

class PostgreSQLConfig(BaseModel):
    """PostgreSQL database configuration."""
    # These fields are loaded from the main Settings object or environment variables
    host: str = Field(..., description="PostgreSQL server host")
    port: int = Field(..., description="PostgreSQL server port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")

    # Connection pool settings for SQLAlchemy
    pool_size: int = Field(default=20, description="Number of connections to keep open in the connection pool")
    max_overflow: int = Field(default=40, description="Number of connections to allow in overflow")
    pool_timeout: int = Field(default=30, description="Timeout in seconds for getting a connection from the pool")
    pool_recycle: int = Field(default=3600, description="Recycle connections after this many seconds")

    @property
    def sync_url(self) -> str:
        """Construct synchronous database URL for SQLAlchemy."""
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_url(self) -> str:
        """Construct asynchronous database URL for SQLAlchemy."""
        # Note: requires 'asyncpg' driver
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class ChromaDBConfig(BaseModel):
    """ChromaDB vector database configuration."""
    # Configuration for server mode, which is used in docker-compose
    host: str = Field(..., description="ChromaDB server host")
    port: int = Field(..., description="ChromaDB server port")

    # Configuration for local/embedded mode (if needed)
    persist_directory: Path = Field(
        default=Path("./chroma_db_data"),
        description="Directory to persist ChromaDB data in local mode"
    )
    
    # Collection settings
    collection_name: str = Field(
        default="legal_documents",
        description="Default name for the main documents collection"
    )

    # Embedding model settings
    embedding_model: str = Field(
        default="paraphrase-multilingual-mpnet-base-v2",
        description="The sentence-transformer model to use for embeddings"
    )

class RedisConfig(BaseModel):
    """Redis configuration for caching."""
    host: str = Field(..., description="Redis server host")
    port: int = Field(..., description="Redis server port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password, if any")

    # Cache settings
    default_ttl: int = Field(default=3600, description="Default Time-To-Live for cache entries in seconds (1 hour)")

    @property
    def url(self) -> str:
        """Construct Redis URL for connection."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

class DatabaseSettings(BaseSettings):
    """
    Main container for all database-related settings.
    Loads values from environment variables.
    """
    # PostgreSQL settings
    postgres_host: str = Field(..., env="POSTGRES_HOST")
    postgres_port: int = Field(..., env="POSTGRES_PORT")
    postgres_db: str = Field(..., env="POSTGRES_DB")
    postgres_user: str = Field(..., env="POSTGRES_USER")
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")

    # ChromaDB settings
    chroma_host: str = Field(..., env="CHROMA_HOST")
    chroma_port: int = Field(..., env="CHROMA_PORT")
    chroma_collection: str = Field("legal_documents", env="CHROMA_COLLECTION")

    # Redis settings
    redis_host: str = Field(..., env="REDIS_HOST")
    redis_port: int = Field(..., env="REDIS_PORT")

    def get_postgres_config(self) -> PostgreSQLConfig:
        return PostgreSQLConfig(
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
            username=self.postgres_user,
            password=self.postgres_password
        )

    def get_chromadb_config(self) -> ChromaDBConfig:
        return ChromaDBConfig(
            host=self.chroma_host,
            port=self.chroma_port,
            collection_name=self.chroma_collection
        )

    def get_redis_config(self) -> RedisConfig:
        return RedisConfig(
            host=self.redis_host,
            port=self.redis_port
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra fields from other config files

# Global instance to be imported by other modules
db_settings = DatabaseSettings()