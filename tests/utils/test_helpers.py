# tests/utils/test_helpers.py
"""
Test helper functions and utilities.
"""
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock

from src.tools.skills_extractor import ExtractedSkill, SkillConfidence, SkillExtractionResult
from src.models.resume import SkillCategory, Skills


def create_mock_skill(
    name: str,
    category: SkillCategory = SkillCategory.TECHNICAL,
    confidence: SkillConfidence = SkillConfidence.HIGH,
    evidence: List[str] = None,
    years_experience: float = None
) -> ExtractedSkill:
    """Create a mock ExtractedSkill for testing."""
    return ExtractedSkill(
        name=name,
        category=category,
        confidence=confidence,
        evidence=evidence or [f"Evidence for {name}"],
        years_experience=years_experience
    )


def create_mock_extraction_result(
    skills: List[ExtractedSkill] = None,
    processing_time: float = 2.0
) -> SkillExtractionResult:
    """Create a mock SkillExtractionResult for testing."""
    if skills is None:
        skills = [
            create_mock_skill("Python", SkillCategory.PROGRAMMING),
            create_mock_skill("React", SkillCategory.FRAMEWORKS),
            create_mock_skill("Leadership", SkillCategory.SOFT_SKILLS)
        ]
    
    # Create categorized skills
    categorized = Skills()
    for skill in skills:
        if skill.category == SkillCategory.PROGRAMMING:
            categorized.programming_languages.append(skill.name)
        elif skill.category == SkillCategory.FRAMEWORKS:
            categorized.frameworks_libraries.append(skill.name)
        elif skill.category == SkillCategory.SOFT_SKILLS:
            categorized.soft_skills.append(skill.name)
        # Add other categories as needed
    
    metadata = {
        "processing_time": processing_time,
        "total_skills_found": len(skills),
        "high_confidence_count": len([s for s in skills if s.confidence == SkillConfidence.HIGH])
    }
    
    return SkillExtractionResult(
        skills=skills,
        categorized_skills=categorized,
        extraction_metadata=metadata
    )


def load_test_data(filename: str) -> Dict[str, Any]:
    """Load test data from JSON file."""
    test_data_path = Path(__file__).parent.parent / "data" / filename
    
    if test_data_path.exists():
        with open(test_data_path, 'r') as f:
            return json.load(f)
    else:
        return {}


def load_sample_text(filename: str) -> str:
    """Load sample text file for testing."""
    samples_path = Path(__file__).parent.parent / "data" / "sample_resumes" / filename
    
    if samples_path.exists():
        with open(samples_path, 'r') as f:
            return f.read()
    else:
        return ""


async def run_async_test(async_func, *args, **kwargs):
    """Helper to run async functions in tests."""
    return await async_func(*args, **kwargs)


def assert_skill_quality(result: SkillExtractionResult, min_skills: int = 5, min_confidence_ratio: float = 0.5):
    """Assert that skill extraction result meets quality standards."""
    assert len(result.skills) >= min_skills, f"Expected at least {min_skills} skills, got {len(result.skills)}"
    
    high_confidence = result.get_high_confidence_skills()
    confidence_ratio = len(high_confidence) / len(result.skills) if result.skills else 0
    
    assert confidence_ratio >= min_confidence_ratio, (
        f"Expected at least {min_confidence_ratio:.1%} high confidence skills, "
        f"got {confidence_ratio:.1%} ({len(high_confidence)}/{len(result.skills)})"
    )


def assert_contains_skills(result: SkillExtractionResult, expected_skills: List[str], min_matches: int = None):
    """Assert that result contains expected skills."""
    if min_matches is None:
        min_matches = len(expected_skills) // 2  # At least half
    
    extracted_names = {skill.name.lower() for skill in result.skills}
    expected_names = {skill.lower() for skill in expected_skills}
    
    matches = expected_names.intersection(extracted_names)
    
    assert len(matches) >= min_matches, (
        f"Expected at least {min_matches} matches from {expected_skills}, "
        f"got {len(matches)}: {list(matches)}. Extracted: {list(extracted_names)}"
    )


# tests/mocks/mock_llm_responses.py
"""
Mock LLM responses for testing without API calls.
"""

MOCK_RESUME_RESPONSE = {
    "skills": [
        {
            "name": "Python",
            "category": "programming_languages",
            "confidence": "high",
            "evidence": ["5+ years experience with Python", "Led development using Python"],
            "years_experience": 5.0,
            "proficiency_level": "Expert",
            "context": "Work experience section"
        },
        {
            "name": "Django",
            "category": "frameworks_libraries",
            "confidence": "high",
            "evidence": ["microservices architecture using Python and Django"],
            "years_experience": None,
            "proficiency_level": "Advanced",
            "context": "Technical implementation"
        },
        {
            "name": "React",
            "category": "frameworks_libraries",
            "confidence": "high",
            "evidence": ["5+ years experience with React and JavaScript"],
            "years_experience": 5.0,
            "proficiency_level": "Expert",
            "context": "Frontend development"
        },
        {
            "name": "Leadership",
            "category": "soft_skills",
            "confidence": "high",
            "evidence": ["Managed team of 8 developers", "Led development"],
            "years_experience": 3.0,
            "proficiency_level": None,
            "context": "Management experience"
        },
        {
            "name": "PostgreSQL",
            "category": "databases",
            "confidence": "medium",
            "evidence": ["REST APIs using Django REST Framework and PostgreSQL"],
            "years_experience": None,
            "proficiency_level": "Intermediate",
            "context": "Database usage"
        },
        {
            "name": "Docker",
            "category": "tools_software",
            "confidence": "medium",
            "evidence": ["CI/CD pipelines with Jenkins and Docker"],
            "years_experience": None,
            "proficiency_level": "Intermediate",
            "context": "DevOps practices"
        },
        {
            "name": "Agile",
            "category": "technical_skills",
            "confidence": "high",
            "evidence": ["team using Agile methodologies"],
            "years_experience": None,
            "proficiency_level": None,
            "context": "Development methodology"
        }
    ]
}

MOCK_JOB_RESPONSE = {
    "skills": [
        {
            "name": "Python",
            "category": "programming_languages",
            "confidence": "high",
            "evidence": ["Expert-level Python programming"],
            "years_experience": 5.0,
            "proficiency_level": "Expert",
            "context": "Required qualification"
        },
        {
            "name": "Django",
            "category": "frameworks_libraries",
            "confidence": "high",
            "evidence": ["Python programming with Django or Flask"],
            "years_experience": None,
            "proficiency_level": "Expert",
            "context": "Required framework"
        },
        {
            "name": "Flask",
            "category": "frameworks_libraries", 
            "confidence": "high",
            "evidence": ["Python programming with Django or Flask"],
            "years_experience": None,
            "proficiency_level": "Expert",
            "context": "Alternative framework"
        },
        {
            "name": "React",
            "category": "frameworks_libraries",
            "confidence": "high",
            "evidence": ["Strong frontend skills with React"],
            "years_experience": None,
            "proficiency_level": "Strong",
            "context": "Required frontend skill"
        },
        {
            "name": "PostgreSQL",
            "category": "databases",
            "confidence": "high",
            "evidence": ["Experience with PostgreSQL and database design"],
            "years_experience": None,
            "proficiency_level": "Experienced",
            "context": "Required database"
        },
        {
            "name": "Docker",
            "category": "tools_software",
            "confidence": "high",
            "evidence": ["Proficiency with Git, Docker, and CI/CD practices"],
            "years_experience": None,
            "proficiency_level": "Proficient",
            "context": "Required tool"
        },
        {
            "name": "AWS",
            "category": "cloud_platforms",
            "confidence": "medium",
            "evidence": ["AWS cloud platform experience"],
            "years_experience": None,
            "proficiency_level": "Experienced",
            "context": "Preferred qualification"
        },
        {
            "name": "TypeScript",
            "category": "programming_languages",
            "confidence": "medium", 
            "evidence": ["TypeScript and modern frontend tooling"],
            "years_experience": None,
            "proficiency_level": "Familiar",
            "context": "Preferred skill"
        },
        {
            "name": "Team Leadership",
            "category": "soft_skills",
            "confidence": "medium",
            "evidence": ["Previous team leadership experience"],
            "years_experience": None,
            "proficiency_level": None,
            "context": "Preferred experience"
        }
    ]
}


# tests/mocks/mock_openai_service.py
"""
Mock OpenAI service for testing without API calls.
"""
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from .mock_llm_responses import MOCK_RESUME_RESPONSE, MOCK_JOB_RESPONSE


class MockOpenAIService:
    """Mock OpenAI service for testing."""
    
    def __init__(self):
        self.llm = Mock()
        self.call_count = 0
        self.last_input = None
    
    def with_structured_output(self, model_class):
        """Mock structured output method."""
        mock_structured = Mock()
        mock_structured.with_structured_output = Mock(return_value=mock_structured)
        return mock_structured
    
    async def mock_extraction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock skill extraction based on context."""
        self.call_count += 1
        self.last_input = input_data
        
        context = input_data.get("context", "resume")
        
        if context == "resume":
            return MOCK_RESUME_RESPONSE
        elif context == "job_description":
            return MOCK_JOB_RESPONSE
        else:
            # Return a simplified response for other contexts
            return {
                "skills": [
                    {
                        "name": "Python",
                        "category": "programming_languages",
                        "confidence": "high",
                        "evidence": ["Python mentioned"],
                        "years_experience": None,
                        "proficiency_level": None,
                        "context": "General context"
                    }
                ]
            }


def create_mock_llm_chain(response_data: Dict[str, Any] = None):
    """Create a mock LLM chain for testing."""
    if response_data is None:
        response_data = MOCK_RESUME_RESPONSE
    
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = response_data
    
    return mock_chain


# tests/utils/assertions.py
"""
Custom assertions for skills testing.
"""

def assert_skill_extraction_quality(result, min_skills=5, min_confidence_ratio=0.6):
    """Assert overall quality of skill extraction."""
    assert len(result.skills) >= min_skills, (
        f"Expected at least {min_skills} skills, got {len(result.skills)}"
    )
    
    high_conf_count = len(result.get_high_confidence_skills())
    confidence_ratio = high_conf_count / len(result.skills)
    
    assert confidence_ratio >= min_confidence_ratio, (
        f"Expected {min_confidence_ratio:.1%} high confidence, "
        f"got {confidence_ratio:.1%} ({high_conf_count}/{len(result.skills)})"
    )


def assert_performance_acceptable(result, max_time=5.0):
    """Assert that performance is within acceptable limits."""
    processing_time = result.extraction_metadata.get("processing_time", 0)
    
    assert processing_time <= max_time, (
        f"Processing took {processing_time:.2f}s, should be <= {max_time}s"
    )
    
    assert processing_time > 0, "Processing time should be recorded"


def assert_skill_categories_present(result, expected_categories, min_categories=3):
    """Assert that expected skill categories are present."""
    found_categories = set()
    
    for skill in result.skills:
        found_categories.add(skill.category)
    
    category_overlap = found_categories.intersection(set(expected_categories))
    
    assert len(category_overlap) >= min_categories, (
        f"Expected at least {min_categories} categories from {expected_categories}, "
        f"found {list(category_overlap)}"
    )