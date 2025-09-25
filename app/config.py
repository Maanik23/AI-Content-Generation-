"""
Configuration management for the AI-powered content creation factory.
Uses Pydantic settings for type-safe configuration with environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    api_title: str = Field(default="AI Content Factory", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    
    # Google Cloud Configuration
    google_project_id: str = Field(default="wmc-automation-agents", description="Google Cloud Project ID")
    google_credentials_path: Optional[str] = Field(default="credentials/google-credentials.json", description="Path to Google Cloud credentials JSON")
    google_drive_folder_id: str = Field(default="1YtN3_CftdJGgK9DFGLSMIky7PbYfFsX5", description="Google Drive folder ID for source documents")
    google_drive_review_folder_id: str = Field(default="1aUwEuIcny7dyLctF-6YFQ28lJkz2PTyK", description="Google Drive folder ID for review documents")
    google_drive_done_folder_id: str = Field(default="1yG_8-wBK1wfrEjzs5J_rKRRaHBpOFPoK", description="Google Drive folder ID for completed content")
    google_sheets_id: str = Field(default="1d87xmQNbWlNwtvRfhaWLSk2FkfTRVadKm94-ppaASbw", description="Google Sheets ID for job tracking")
    
    # AI Model Configuration - Gemini 2.5 Pro
    gemini_model_name: str = Field(default="gemini-2.0-flash-exp", description="Gemini model name (using 2.0 flash exp as 2.5 pro is not yet available)")
    gemini_api_key: str = Field(default="your-gemini-api-key", description="Gemini API key")
    imagen_model_name: str = Field(default="imagegeneration@006", description="Imagen 2 model name")
    
    # Google Services Configuration
    google_application_credentials_json: Optional[str] = Field(default=None, description="Google service account JSON credentials")
    google_drive_api_version: str = Field(default="v3", description="Google Drive API version")
    google_sheets_api_version: str = Field(default="v4", description="Google Sheets API version")
    google_docs_api_version: str = Field(default="v1", description="Google Docs API version")
    
    # ElevenLabs Configuration
    elevenlabs_api_key: str = Field(default="your-elevenlabs-api-key", description="ElevenLabs API key")
    elevenlabs_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID")
    
    # File Processing Configuration
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB (increased for large documents)")
    supported_file_types: list[str] = Field(default=[".docx"], description="Supported file types")
    temp_dir: str = Field(default="/tmp", description="Temporary directory for file processing")
    
    # Large Document Processing
    max_document_words: int = Field(default=50000, description="Maximum document words to process")
    chunk_size: int = Field(default=10000, description="Chunk size for processing large documents")
    enable_chunking: bool = Field(default=True, description="Enable chunking for very large documents")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format string"
    )
    
    # Retry Configuration
    max_retries: int = Field(default=3, description="Maximum number of retries for API calls")
    retry_delay: float = Field(default=1.0, description="Initial retry delay in seconds")
    retry_backoff_factor: float = Field(default=2.0, description="Retry backoff factor")
    
    # Language Configuration
    default_language: str = Field(default="de", description="Default language for content generation (de = German)")
    output_language: str = Field(default="de", description="Output language for all generated content")
    
    # Professional Production Configuration
    enable_monitoring: bool = Field(default=True, description="Enable professional monitoring and logging")
    enable_audit_logs: bool = Field(default=True, description="Enable audit logging for security")
    enable_access_logs: bool = Field(default=True, description="Enable access logging")
    enable_data_logs: bool = Field(default=True, description="Enable data processing logs")
    
    # Scalability Configuration
    max_concurrent_jobs: int = Field(default=5, description="Maximum concurrent processing jobs")
    queue_size: int = Field(default=100, description="Maximum queue size for pending jobs")
    timeout_seconds: int = Field(default=300, description="Timeout for job processing in seconds")
    
    # Cost Control Configuration
    budget_limit: float = Field(default=10.0, description="Monthly budget limit in EUR")
    budget_currency: str = Field(default="EUR", description="Budget currency")
    cost_alert_threshold: float = Field(default=8.0, description="Cost alert threshold (80% of budget)")
    auto_stop_services: bool = Field(default=True, description="Automatically stop services if budget exceeded")
    
    # Performance Configuration
    cache_ttl: int = Field(default=3600, description="Cache time-to-live in seconds")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    enable_compression: bool = Field(default=True, description="Enable response compression")
    
    # n8n Integration Configuration
    n8n_user: str = Field(default="", description="n8n username")
    n8n_password: str = Field(default="", description="n8n password")
    n8n_encryption_key: str = Field(default="", description="n8n encryption key")
    n8n_postgres_password: str = Field(default="", description="n8n PostgreSQL password")
    n8n_url: str = Field(default="http://localhost:5678", description="n8n URL")
    ai_content_factory_url: str = Field(default="http://localhost:8000", description="AI Content Factory URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from environment


# Global settings instance
settings = Settings()
