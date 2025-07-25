# tests/unit/test_skill_models.py
"""
Unit tests for skill-related Pydantic models.
"""
import pytest
from src.tools.skills_extractor import ExtractedSkill, SkillExtractionResult, SkillConfidence
from src.models.resume import SkillCategory, Skills


class TestExtractedSkill:
    """Test ExtractedSkill model validation and behavior."""
    
    def test_skill_standardization(self):
        """Test that skill names are standardized correctly."""
        test_cases = [
            ("js", "JavaScript"),
            ("typescript", "TypeScript"), 
            ("react.js", "React"),
            ("node.js", "Node.js"),
            ("postgresql", "PostgreSQL"),
            ("ml", "Machine Learning"),
            ("ai", "Artificial Intelligence"),
            ("aws", "Amazon Web Services")
        ]
        
        for input_name, expected_name in test_cases:
            skill = ExtractedSkill(
                name=input_name,
                category=SkillCategory.PROGRAMMING,
                confidence=SkillConfidence.HIGH,
                evidence=["test evidence"]
            )
            assert skill.name == expected_name, f"Expected {expected_name}, got {skill.name}"
    
    def test_skill_validation(self):
        """Test that ExtractedSkill validates required fields."""
        # Valid skill
        skill = ExtractedSkill(
            name="Python",
            category=SkillCategory.PROGRAMMING,
            confidence=SkillConfidence.HIGH,
            evidence=["5 years Python experience"]
        )
        assert skill.name == "Python"
        assert skill.category == SkillCategory.PROGRAMMING
        assert skill.confidence == SkillConfidence.HIGH
        
        # Test with optional fields
        skill_with_experience = ExtractedSkill(
            name="Django",
            category=SkillCategory.FRAMEWORKS,
            confidence=SkillConfidence.MEDIUM,
            evidence=["Built Django apps"],
            years_experience=3.0,
            proficiency_level="Intermediate"
        )
        assert skill_with_experience.years_experience == 3.0
        assert skill_with_experience.proficiency_level == "Intermediate"


class TestSkillExtractionResult:
    """Test SkillExtractionResult helper methods."""
    
    @pytest.fixture
    def sample_result(self, sample_extracted_skills):
        """Create a sample SkillExtractionResult for testing."""
        categorized_skills = Skills(
            programming_languages=["Python"],
            frameworks_libraries=["Django"],
            tools_software=["Docker"],
            soft_skills=["Leadership"]
        )
        
        return SkillExtractionResult(
            skills=sample_extracted_skills,
            categorized_skills=categorized_skills,
            extraction_metadata={
                "processing_time": 2.5,
                "total_skills_found": 4,
                "high_confidence_count": 2
            }
        )
    
    def test_get_skills_by_category(self, sample_result):
        """Test filtering skills by category."""
        # Test programming languages
        programming_skills = sample_result.get_skills_by_category(SkillCategory.PROGRAMMING)
        assert len(programming_skills) == 1
        assert programming_skills[0].name == "Python"
        
        # Test soft skills
        soft_skills = sample_result.get_skills_by_category(SkillCategory.SOFT_SKILLS)
        assert len(soft_skills) == 1
        assert soft_skills[0].name == "Leadership"
        
        # Test non-existent category
        domain_skills = sample_result.get_skills_by_category(SkillCategory.DOMAIN)
        assert len(domain_skills) == 0
    
    def test_get_high_confidence_skills(self, sample_result):
        """Test filtering high confidence skills."""
        high_confidence = sample_result.get_high_confidence_skills()
        assert len(high_confidence) == 2
        
        skill_names = {skill.name for skill in high_confidence}
        assert "Python" in skill_names
        assert "Leadership" in skill_names
        assert "Django" not in skill_names  # Medium confidence
        assert "Docker" not in skill_names  # Low confidence
    
    def test_get_skills_with_experience(self, sample_result):
        """Test filtering skills with experience data."""
        experienced_skills = sample_result.get_skills_with_experience()
        assert len(experienced_skills) == 2
        
        skill_names = {skill.name for skill in experienced_skills}
        assert "Python" in skill_names
        assert "Leadership" in skill_names
        
        # Check experience values
        python_skill = next(s for s in experienced_skills if s.name == "Python")
        assert python_skill.years_experience == 5.0


# tests/unit/test_skill_validation.py
"""
Unit tests for skill validation logic.
"""
import pytest
from src.tools.skills_extractor import LLMSkillExtractor


class TestSkillValidation:
    """Test skill validation and categorization logic."""
    
    def test_knowledge_base_loading(self):
        """Test that skill knowledge base loads correctly."""
        extractor = LLMSkillExtractor()
        kb = extractor.skill_knowledge_base
        
        # Check that all expected categories exist
        expected_categories = [
            "programming_languages",
            "frameworks_libraries", 
            "databases",
            "cloud_platforms",
            "tools_software",
            "technical_skills",
            "soft_skills"
        ]
        
        for category in expected_categories:
            assert category in kb, f"Missing category: {category}"
            assert len(kb[category]) > 0, f"Empty category: {category}"
        
        # Check some specific skills
        assert "Python" in kb["programming_languages"]
        assert "React" in kb["frameworks_libraries"]
        assert "PostgreSQL" in kb["databases"]
        assert "AWS" in kb["cloud_platforms"]
    
    def test_context_specific_instructions(self):
        """Test that context-specific instructions are different."""
        extractor = LLMSkillExtractor()
        
        resume_instructions = extractor._get_context_specific_instructions("resume")
        job_instructions = extractor._get_context_specific_instructions("job_description")
        project_instructions = extractor._get_context_specific_instructions("project_description")
        
        # Each should be different
        assert resume_instructions != job_instructions
        assert job_instructions != project_instructions
        assert resume_instructions != project_instructions
        
        # Check for context-specific keywords
        assert "DEMONSTRATED" in resume_instructions
        assert "work experience" in resume_instructions.lower()
        
        assert "REQUIRED vs PREFERRED" in job_instructions
        assert "must have" in job_instructions.lower()
        
        assert "TECHNICAL STACK" in project_instructions
        assert "implementation" in project_instructions.lower()
    
    def test_context_specific_prompts(self):
        """Test context-specific human prompts."""
        extractor = LLMSkillExtractor()
        
        resume_prompt = extractor._get_context_specific_human_prompt("resume")
        job_prompt = extractor._get_context_specific_human_prompt("job_description")
        project_prompt = extractor._get_context_specific_human_prompt("project_description")
        
        # Each should be different
        assert resume_prompt != job_prompt
        assert job_prompt != project_prompt
        
        # Check for context-specific content
        assert "RESUME ANALYSIS FOCUS" in resume_prompt
        assert "JOB REQUIREMENTS ANALYSIS" in job_prompt
        assert "PROJECT TECHNOLOGY ANALYSIS" in project_prompt


# tests/unit/test_skill_categorization.py
"""
Unit tests for skill categorization logic.
"""
import pytest
from src.tools.skills_extractor import LLMSkillExtractor, ExtractedSkill, SkillConfidence
from src.models.resume import SkillCategory, Skills


class TestSkillCategorization:
    """Test skill categorization functionality."""
    
    def test_categorize_skills(self):
        """Test conversion from ExtractedSkills to Skills model."""
        extractor = LLMSkillExtractor()
        
        extracted_skills = [
            ExtractedSkill(
                name="Python",
                category=SkillCategory.PROGRAMMING,
                confidence=SkillConfidence.HIGH,
                evidence=["Python development"]
            ),
            ExtractedSkill(
                name="React",
                category=SkillCategory.FRAMEWORKS,
                confidence=SkillConfidence.HIGH,
                evidence=["React applications"]
            ),
            ExtractedSkill(
                name="PostgreSQL",
                category=SkillCategory.DATABASES,
                confidence=SkillConfidence.MEDIUM,
                evidence=["PostgreSQL database"]
            ),
            ExtractedSkill(
                name="Leadership",
                category=SkillCategory.SOFT_SKILLS,
                confidence=SkillConfidence.HIGH,
                evidence=["Team leadership"]
            ),
            ExtractedSkill(
                name="Docker",
                category=SkillCategory.TOOLS,
                confidence=SkillConfidence.MEDIUM,
                evidence=["Docker containers"]
            )
        ]
        
        categorized = extractor._categorize_skills(extracted_skills)
        
        # Check that skills are properly categorized
        assert "Python" in categorized.programming_languages
        assert "React" in categorized.frameworks_libraries
        assert "PostgreSQL" in categorized.databases
        assert "Leadership" in categorized.soft_skills
        assert "Docker" in categorized.tools_software
        
        # Check that categories are not cross-contaminated
        assert len(categorized.programming_languages) == 1
        assert len(categorized.frameworks_libraries) == 1
        assert len(categorized.databases) == 1
        assert len(categorized.soft_skills) == 1
        assert len(categorized.tools_software) == 1
    
    def test_skills_model_helper_methods(self):
        """Test Skills model helper methods."""
        skills = Skills(
            programming_languages=["Python", "JavaScript"],
            frameworks_libraries=["Django", "React"],
            databases=["PostgreSQL", "MongoDB"],
            soft_skills=["Leadership", "Communication"]
        )
        
        # Test get_all_skills
        all_skills = skills.get_all_skills()
        expected_skills = {
            "Python", "JavaScript", "Django", "React", 
            "PostgreSQL", "MongoDB", "Leadership", "Communication"
        }
        assert set(all_skills) == expected_skills
        
        # Test get_technical_skills (excludes soft skills)
        technical_skills = skills.get_technical_skills()
        expected_technical = {
            "Python", "JavaScript", "Django", "React", "PostgreSQL", "MongoDB"
        }
        assert set(technical_skills) == expected_technical
        assert "Leadership" not in technical_skills
        assert "Communication" not in technical_skills