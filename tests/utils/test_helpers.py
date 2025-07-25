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

