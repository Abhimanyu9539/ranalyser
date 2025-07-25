"""
Shared pytest fixtures and configuration for all tests.
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.skills_extractor import LLMSkillExtractor, ExtractedSkill, SkillConfidence
from src.models.resume import SkillCategory, Skills


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def skill_extractor():
    """Create a skill extractor instance for testing."""
    return LLMSkillExtractor()


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing."""
    return """
    Senior Software Engineer
    john.doe@email.com | (555) 123-4567
    
    EXPERIENCE:
    Senior Software Engineer at TechCorp (2020-Present)
    • Led development of microservices architecture using Python and Django
    • Managed team of 8 developers using Agile methodologies  
    • Implemented CI/CD pipelines with Jenkins and Docker
    • 5+ years experience with React and JavaScript frontend development
    
    Software Engineer at StartupXYZ (2018-2020)
    • Built RESTful APIs using Node.js and Express
    • Developed mobile applications with React Native
    • Used AWS services including EC2, S3, and Lambda
    
    SKILLS:
    Programming: Python, JavaScript, TypeScript, Java
    Frameworks: Django, React, Node.js, Express
    Databases: PostgreSQL, MongoDB, Redis
    Cloud: AWS, Docker, Kubernetes
    """


@pytest.fixture
def sample_job_description():
    """Sample job description for testing."""
    return """
    Senior Full Stack Developer - Remote
    
    REQUIRED QUALIFICATIONS:
    • 5+ years of professional software development experience
    • Expert-level Python programming with Django or Flask
    • Strong frontend skills with React and modern JavaScript
    • Experience with PostgreSQL and database design
    • Proficiency with Git, Docker, and CI/CD practices
    
    PREFERRED QUALIFICATIONS:
    • AWS cloud platform experience
    • TypeScript and modern frontend tooling
    • Experience with microservices architecture
    • Previous team leadership experience
    """


@pytest.fixture
def sample_extracted_skills():
    """Sample list of extracted skills for testing."""
    return [
        ExtractedSkill(
            name="Python",
            category=SkillCategory.PROGRAMMING,
            confidence=SkillConfidence.HIGH,
            evidence=["5+ years experience", "Led development using Python"],
            years_experience=5.0,
            proficiency_level="Expert"
        ),
        ExtractedSkill(
            name="Leadership",
            category=SkillCategory.SOFT_SKILLS,
            confidence=SkillConfidence.HIGH,
            evidence=["Managed team of 8 developers"],
            years_experience=3.0
        ),
        ExtractedSkill(
            name="Django",
            category=SkillCategory.FRAMEWORKS,
            confidence=SkillConfidence.MEDIUM,
            evidence=["microservices architecture using Python and Django"]
        ),
        ExtractedSkill(
            name="Docker",
            category=SkillCategory.TOOLS,
            confidence=SkillConfidence.LOW,
            evidence=["CI/CD pipelines with Jenkins and Docker"]
        )
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing."""
    return {
        "skills": [
            {
                "name": "Python",
                "category": "programming_languages",
                "confidence": "high",
                "evidence": ["5+ years experience with Python"],
                "years_experience": 5.0,
                "proficiency_level": "Expert",
                "context": "Work experience"
            },
            {
                "name": "Leadership",
                "category": "soft_skills",
                "confidence": "high", 
                "evidence": ["Managed team of 8 developers"],
                "years_experience": None,
                "proficiency_level": None,
                "context": "Management experience"
            }
        ]
    }


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_resumes_dir(test_data_dir):
    """Get path to sample resumes directory."""
    return test_data_dir / "sample_resumes"


@pytest.fixture
def sample_jobs_dir(test_data_dir):
    """Get path to sample job descriptions directory."""
    return test_data_dir / "sample_jobs"


@pytest.fixture
def skip_if_no_api_key():
    """Skip test if OpenAI API key is not available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for unit tests."""
    mock_service = Mock()
    mock_service.llm = Mock()
    
    # Mock structured output
    mock_structured_llm = Mock()
    mock_service.llm.with_structured_output.return_value = mock_structured_llm
    
    return mock_service


@pytest.fixture
async def async_mock_llm_service():
    """Async mock LLM service for testing."""
    mock_service = Mock()
    mock_chain = AsyncMock()
    
    # Set up the mock chain
    mock_service.llm.with_structured_output.return_value = mock_service.llm
    mock_service.llm.__or__ = lambda self, other: mock_chain
    
    return mock_service, mock_chain


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "unit: Unit tests that don't require external APIs"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests that require API keys"
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that may take longer to run"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark tests in integration/ directory as integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)