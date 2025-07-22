"""
Configuration settings for the Resume Analyzer application.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    
    # LangChain Configuration
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="resume-analyzer", env="LANGCHAIN_PROJECT")
    
    # Job Board APIs
    linkedin_api_key: Optional[str] = Field(default=None, env="LINKEDIN_API_KEY")
    indeed_publisher_id: Optional[str] = Field(default=None, env="INDEED_PUBLISHER_ID")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./resume_analyzer.db", env="DATABASE_URL")
    
    # Application Settings
    debug: bool = Field(default=False, env="DEBUG")
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(
        default=["pdf", "docx", "txt"], 
        env="ALLOWED_EXTENSIONS"
    )
    host: str = Field(default="localhost", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Storage Paths
    upload_folder: str = Field(default="frontend/static/uploads", env="UPLOAD_FOLDER")
    output_folder: str = Field(default="data/outputs", env="OUTPUT_FOLDER")
    
    # Resume Analysis Settings
    max_resume_sections: int = Field(default=20, env="MAX_RESUME_SECTIONS")
    min_ats_score: float = Field(default=60.0, env="MIN_ATS_SCORE")
    max_job_matches: int = Field(default=10, env="MAX_JOB_MATCHES")
    
    # Similarity Thresholds
    skill_similarity_threshold: float = Field(default=0.75, env="SKILL_SIMILARITY_THRESHOLD")
    experience_similarity_threshold: float = Field(default=0.70, env="EXPERIENCE_SIMILARITY_THRESHOLD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    


# Global settings instance
settings = Settings()


# LangChain environment setup
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2)
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project    