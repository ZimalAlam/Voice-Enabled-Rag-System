from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Keys
    gemini_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None
    google_cloud_service_account_path: Optional[str] = None
    serp_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Primary AI provider configuration
    ai_provider: str = "gemini"  # gemini or claude
    
    # Google Drive OAuth
    google_drive_client_id: Optional[str] = None
    google_drive_client_secret: Optional[str] = None
    google_drive_redirect_uri: str = "http://localhost:8000/auth/google/callback"
    
    # Database Configuration
    vector_db_type: str = "chromadb"  # chromadb, qdrant, weaviate
    vector_db_url: Optional[str] = None
    vector_db_port: Optional[int] = None
    chroma_persist_directory: str = "./chroma_db"
    
    # Redis Configuration (for caching and queues)
    redis_url: str = "redis://localhost:6379"
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Gemini model configuration
    gemini_chat_model: str = "gemini-2.0-flash-lite-001"
    gemini_vision_model: str = "gemini-2.0-flash-lite-001"
    
    # Claude model configuration  
    claude_chat_model: str = "claude-3-sonnet-20240229"
    claude_vision_model: str = "claude-3-sonnet-20240229"
    
    # STT Configuration
    stt_provider: str = "google"  # google (Cloud Speech-to-Text)
    google_speech_model: str = "latest_long"
    
    # Document Processing
    max_file_size_mb: int = 100
    supported_file_types: str = ".pdf,.docx,.txt,.md"
    ocr_enabled: bool = True
    extract_images: bool = True
    
    @property
    def supported_file_types_list(self) -> List[str]:
        """Get supported file types as a list"""
        return [ext.strip() for ext in self.supported_file_types.split(",")]
    
    # Search Configuration
    max_search_results: int = 10
    similarity_threshold: float = 0.7
    max_tokens_per_chunk: int = 1000
    chunk_overlap: int = 200
    
    # API Configuration
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Development
    debug: bool = False
    reload: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Validate critical settings
def validate_settings():
    """Validate that required settings are present"""
    if settings.ai_provider == "gemini" and not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required when using Gemini as AI provider")
    
    if settings.ai_provider == "claude" and not settings.claude_api_key:
        raise ValueError("CLAUDE_API_KEY is required when using Claude as AI provider")
    
    if settings.ai_provider not in ["gemini", "claude"]:
        raise ValueError("AI_PROVIDER must be either 'gemini' or 'claude'")
    
    if settings.vector_db_type not in ["chromadb", "qdrant", "weaviate"]:
        raise ValueError("Invalid vector_db_type. Must be one of: chromadb, qdrant, weaviate")
    
    if settings.vector_db_type != "chromadb" and not settings.vector_db_url:
        raise ValueError(f"VECTOR_DB_URL is required when using {settings.vector_db_type}")

# Run validation on import
validate_settings() 