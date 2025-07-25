# tests/integration/test_skill_extraction.py
"""
Integration tests for skill extraction with real LLM calls.
"""
import pytest
from src.tools.skills_extractor import (
    LLMSkillExtractor, 
    extract_skills_from_resume,
    extract_skills_from_job,
    compare_resume_job_skills
)


@pytest.mark.integration
class TestRealSkillExtraction:
    """Integration tests with real OpenAI API calls."""
    
    @pytest.mark.asyncio
    async def test_resume_extraction(self, skill_extractor, sample_resume_text, skip_if_no_api_key):
        """Test real resume skill extraction."""
        result = await skill_extractor.extract_skills_from_resume(sample_resume_text)
        
        # Basic validation
        assert len(result.skills) > 0, "Should extract at least some skills"
        assert len(result.categorized_skills.get_all_skills()) > 0, "Should categorize skills"
        
        # Check for expected skills from the sample resume
        skill_names = {skill.name.lower() for skill in result.skills}
        expected_skills = {"python", "javascript", "django", "react", "postgresql"}
        
        found_expected = expected_skills.intersection(skill_names)
        assert len(found_expected) >= 3, f"Expected to find at least 3 of {expected_skills}, found: {skill_names}"
        
        # Quality checks
        high_confidence = result.get_high_confidence_skills()
        assert len(high_confidence) >= len(result.skills) * 0.5, "At least 50% should be high confidence"
        
        # Performance check
        assert result.extraction_metadata["processing_time"] < 10.0, "Should complete within 10 seconds"
        
        print(f"✅ Extracted {len(result.skills)} skills in {result.extraction_metadata['processing_time']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_job_description_extraction(self, skill_extractor, sample_job_description, skip_if_no_api_key):
        """Test real job description skill extraction."""
        result = await skill_extractor.extract_skills_from_job_description(sample_job_description)
        
        assert len(result.skills) > 0, "Should extract job requirements"
        
        # Check for expected job skills
        skill_names = {skill.name.lower() for skill in result.skills}
        expected_job_skills = {"python", "django", "flask", "react", "postgresql", "docker"}
        
        found_expected = expected_job_skills.intersection(skill_names)
        assert len(found_expected) >= 4, f"Expected job skills: {expected_job_skills}, found: {skill_names}"
        
        print(f"✅ Extracted {len(result.skills)} job requirements")
    
    @pytest.mark.asyncio
    async def test_skill_comparison(self, sample_resume_text, sample_job_description, skip_if_no_api_key):
        """Test skill comparison between resume and job."""
        comparison = await compare_resume_job_skills(sample_resume_text, sample_job_description)
        
        assert "resume_skills" in comparison
        assert "job_skills" in comparison
        assert "comparison" in comparison
        
        comp_data = comparison["comparison"]
        
        # Validate comparison structure
        required_fields = ["match_percentage", "matched_skills", "missing_skills", "additional_skills"]
        for field in required_fields:
            assert field in comp_data, f"Missing field: {field}"
        
        # Validate data types and ranges
        assert 0 <= comp_data["match_percentage"] <= 100, "Match percentage should be 0-100"
        assert isinstance(comp_data["matched_skills"], list), "Matched skills should be a list"
        assert isinstance(comp_data["missing_skills"], list), "Missing skills should be a list"
        
        print(f"✅ Match: {comp_data['match_percentage']}%, "
              f"Matched: {len(comp_data['matched_skills'])}, "
              f"Missing: {len(comp_data['missing_skills'])}")


# tests/integration/test_context_behavior.py
"""
Integration tests for context-specific behavior.
"""
import pytest
from src.tools.skills_extractor import LLMSkillExtractor


@pytest.mark.integration 
class TestContextBehavior:
    """Test that different contexts produce different results."""
    
    @pytest.mark.asyncio
    async def test_leadership_context_differences(self, skill_extractor, skip_if_no_api_key):
        """Test that leadership is interpreted differently in different contexts."""
        text = "Led team of 8 developers using Agile methodologies"
        
        # Extract from resume context
        resume_result = await skill_extractor.extract_skills_from_text(text, "resume")
        
        # Extract from job description context  
        job_result = await skill_extractor.extract_skills_from_text(text, "job_description")
        
        # Both should find leadership skills
        resume_skills = {skill.name.lower() for skill in resume_result.skills}
        job_skills = {skill.name.lower() for skill in job_result.skills}
        
        leadership_terms = {"leadership", "team leadership", "management"}
        
        resume_has_leadership = any(term in resume_skills for term in leadership_terms)
        job_has_leadership = any(term in job_skills for term in leadership_terms)
        
        assert resume_has_leadership, f"Resume context should find leadership in: {resume_skills}"
        assert job_has_leadership, f"Job context should find leadership in: {job_skills}"
        
        print(f"Resume skills: {resume_skills}")
        print(f"Job skills: {job_skills}")
    
    @pytest.mark.asyncio
    async def test_technical_stack_contexts(self, skill_extractor, skip_if_no_api_key):
        """Test technical stack extraction across contexts."""
        text = "Built using Python Django framework with PostgreSQL database and deployed on AWS"
        
        contexts = ["resume", "job_description", "project_description"]
        results = {}
        
        for context in contexts:
            result = await skill_extractor.extract_skills_from_text(text, context)
            results[context] = {skill.name.lower() for skill in result.skills}
        
        expected_tech_skills = {"python", "django", "postgresql", "aws"}
        
        # All contexts should find core technical skills
        for context, skills in results.items():
            found_tech = expected_tech_skills.intersection(skills)
            assert len(found_tech) >= 3, f"Context '{context}' should find technical skills: {skills}"
        
        print("Context-specific results:")
        for context, skills in results.items():
            print(f"  {context}: {skills}")


# tests/integration/test_performance.py
"""
Performance tests for skill extraction.
"""
import pytest
import asyncio
import time
from src.tools.skills_extractor import LLMSkillExtractor


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Performance and scalability tests."""
    
    @pytest.mark.asyncio
    async def test_extraction_performance(self, skill_extractor, sample_resume_text, skip_if_no_api_key):
        """Test that extraction meets performance requirements."""
        start_time = time.time()
        
        result = await skill_extractor.extract_skills_from_resume(sample_resume_text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 5.0, f"Processing took {processing_time:.2f}s, should be < 5s"
        assert len(result.skills) >= 5, f"Should extract at least 5 skills, got {len(result.skills)}"
        
        # Quality assertions
        high_confidence_ratio = len(result.get_high_confidence_skills()) / len(result.skills)
        assert high_confidence_ratio >= 0.5, f"High confidence ratio {high_confidence_ratio:.2f} too low"
        
        print(f"⚡ Performance: {processing_time:.2f}s, {len(result.skills)} skills, "
              f"{high_confidence_ratio:.1%} high confidence")
    
    @pytest.mark.asyncio
    async def test_concurrent_extractions(self, skill_extractor, sample_resume_text, skip_if_no_api_key):
        """Test concurrent skill extractions."""
        num_concurrent = 3
        
        async def extract_skills():
            return await skill_extractor.extract_skills_from_resume(sample_resume_text)
        
        start_time = time.time()
        
        # Run concurrent extractions
        tasks = [extract_skills() for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check that most succeeded
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= num_concurrent * 0.8, "Most concurrent requests should succeed"
        
        # Check performance
        avg_time_per_request = total_time / num_concurrent
        assert avg_time_per_request < 10.0, f"Average time per request: {avg_time_per_request:.2f}s"
        
        print(f"🔄 Concurrent test: {len(successful_results)}/{num_concurrent} succeeded, "
              f"{avg_time_per_request:.2f}s average")
    
    @pytest.mark.asyncio
    async def test_large_text_handling(self, skill_extractor, skip_if_no_api_key):
        """Test handling of large text inputs."""
        # Create a large resume text
        large_text = """
        Senior Software Engineer with extensive experience.
        
        TECHNICAL SKILLS:
        """ + "Python developer with Django experience. " * 100 + """
        JavaScript and React development experience.
        PostgreSQL database administration.
        AWS cloud platform expertise.
        Docker containerization knowledge.
        """
        
        try:
            result = await skill_extractor.extract_skills_from_resume(large_text)
            
            # Should handle large text gracefully
            assert len(result.skills) > 0, "Should extract skills from large text"
            assert result.extraction_metadata["processing_time"] < 15.0, "Should handle large text efficiently"
            
            print(f"📄 Large text: {len(large_text)} chars, {len(result.skills)} skills extracted")
            
        except Exception as e:
            # If it fails, should fail gracefully
            assert "timeout" in str(e).lower() or "limit" in str(e).lower(), f"Unexpected error: {e}"
            print(f"⚠️  Large text handling failed gracefully: {e}")
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, skill_extractor, skip_if_no_api_key):
        """Test edge cases and error handling."""
        edge_cases = [
            ("minimal_text", "Python developer"),
            ("no_skills", "This is just regular text with no technical content"),
            ("special_chars", "Développeur Python avec expérience en Django™ et React®"),
            ("abbreviations", "ML engineer with DL and NLP experience using TF and PyTorch")
        ]
        
        for case_name, text in edge_cases:
            try:
                result = await skill_extractor.extract_skills_from_text(text, "resume")
                
                print(f"✅ {case_name}: {len(result.skills)} skills extracted")
                
                # Basic validation
                assert isinstance(result.skills, list), f"{case_name}: should return list"
                assert result.extraction_metadata["processing_time"] > 0, f"{case_name}: should have processing time"
                
            except Exception as e:
                print(f"⚠️  {case_name} failed: {e}")
                # Edge cases can fail, but should fail gracefully
                assert "SkillExtractionError" in str(type(e)), f"Should raise SkillExtractionError, got {type(e)}"