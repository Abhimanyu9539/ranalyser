"""
Configuration settings for the Resume Analyzer application.
"""
import os
from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


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
    allowed_extensions: List[str] = Field(default=[".pdf", ".docx", ".txt"])
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
    
    @field_validator('allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from string or list."""
        if isinstance(v, str):
            # Handle comma-separated string
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["pdf", "docx", "txt"]  # Default fallback
    
    @field_validator('langchain_tracing_v2', mode='before')
    @classmethod
    def parse_bool(cls, v):
        """Parse boolean from string."""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # Allow extra fields from environment
        extra = "ignore"


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"Warning: Could not load settings properly: {e}")
    print("Using default settings for development...")
    
    # Create minimal settings for development
    class DefaultSettings:
        def __init__(self):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_model = "gpt-4"
            self.openai_temperature = 0.1
            self.langchain_api_key = None
            self.langchain_tracing_v2 = False
            self.langchain_project = "resume-analyzer"
            self.linkedin_api_key = None
            self.indeed_publisher_id = None
            self.database_url = "sqlite:///./resume_analyzer.db"
            self.debug = True
            self.max_file_size = 10485760
            self.allowed_extensions = ["pdf", "docx", "txt"]
            self.host = "localhost"
            self.port = 8000
            self.upload_folder = "frontend/static/uploads"
            self.output_folder = "data/outputs"
            self.max_resume_sections = 20
            self.min_ats_score = 60.0
            self.max_job_matches = 10
            self.skill_similarity_threshold = 0.75
            self.experience_similarity_threshold = 0.70
    
    settings = DefaultSettings()


# LangChain environment setup
if hasattr(settings, 'langchain_tracing_v2') and settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2)
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project