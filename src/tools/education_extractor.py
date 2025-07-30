"""
LLM-based work experience extraction from resume text.
Uses OpenAI/LangChain with function calling for reliable extraction of work history.
Preserves original date formats exactly as written in the resume.
"""
import logging
import time
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from src.models.resume import WorkExperience
from src.services.openai_service import langgraph_openai_service, OpenAIServiceError
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkExperienceExtractionError(Exception):
    """Custom exception for work experience extraction errors."""
    pass


class WorkExperienceExtractionResult(BaseModel):
    """Work experience extraction result with metadata."""
    experiences: List[WorkExperience] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_total_years_experience(self) -> float:
        """Calculate total years of experience across all positions."""
        total_years = 0.0
        current_year = datetime.now().year
        current_month = datetime.now().month

        for exp in self.experiences:
            try:
                # Check if dates have month precision
                start_has_month = '/' in exp.start_date
                end_has_month = '/' in exp.end_date and exp.end_date.lower() != 'present'
                is_present = exp.end_date.lower() == 'present'
                
                if start_has_month or end_has_month or is_present:
                    # Use month-level calculation
                    if start_has_month:
                        start_parts = exp.start_date.split('/')
                        start_month, start_year = int(start_parts[0]), int(start_parts[1])
                    else:
                        start_year = int(exp.start_date)
                        start_month = 1
                    
                    if is_present:
                        end_month, end_year = current_month, current_year
                    elif end_has_month:
                        end_parts = exp.end_date.split('/')
                        end_month, end_year = int(end_parts[0]), int(end_parts[1])
                    else:
                        end_year = int(exp.end_date)
                        end_month = 12
                    
                    months = ((end_year - start_year) * 12) + (end_month - start_month)
                    total_years += max(0, months / 12)
                else:
                    # Both dates are year-only - simple year calculation
                    start_year = int(exp.start_date)
                    end_year = int(exp.end_date)
                    years = end_year - start_year + 1
                    total_years += max(0, years)
                    
            except (ValueError, IndexError):
                continue
        
        return round(total_years, 1)
    
    def get_current_position(self) -> Optional[WorkExperience]:
        """Get the current position (end_date = 'Present')."""
        for exp in self.experiences:
            if exp.end_date.lower() == 'present':
                return exp
        return None
    
    def get_companies_worked_at(self) -> List[str]:
        """Get list of unique companies."""
        return list(set(exp.company for exp in self.experiences))
    
    def get_most_recent_position(self) -> Optional[WorkExperience]:
        """Get the most recent position."""
        if not self.experiences:
            return None
        return self.experiences[0]  # Already sorted by date


class WorkExperienceLLMResult(BaseModel):
    """Wrapper for LLM extraction result."""
    experiences: List[WorkExperience] = Field(default_factory=list)


class LLMWorkExperienceExtractor:
    """LLM-based work experience extraction system."""
    
    def __init__(self):
        """Initialize the work experience extractor."""
        self.openai_service = langgraph_openai_service
        logger.info("LLM Work Experience Extractor initialized")
    
    async def extract_work_experience_from_text(self, text: str) -> WorkExperienceExtractionResult:
        """
        Extract work experience from resume text using LLM.
        
        Args:
            text: The resume text to extract work experience from
        
        Returns:
            WorkExperienceExtractionResult with extracted experience data and metadata
        """
        start_time = time.time()
        
        try:
            logger.info("Starting LLM work experience extraction")
            
            # Perform LLM extraction
            experiences = await self._perform_llm_extraction(text)
            
            # Post-process experiences
            processed_experiences = self._post_process_experiences(experiences)
            
            # Calculate metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time": processing_time,
                "total_positions": len(processed_experiences),
                "current_positions": len([exp for exp in processed_experiences if exp.end_date.lower() == 'present']),
                "companies": list(set(exp.company for exp in processed_experiences)),
                "total_years_experience": self._calculate_total_years(processed_experiences),
                "model_used": settings.openai_model,
                "extraction_method": "llm_function_calling"
            }
            
            result = WorkExperienceExtractionResult(
                experiences=processed_experiences,
                extraction_metadata=metadata
            )
            
            logger.info(f"Work experience extraction completed. Found {len(processed_experiences)} positions in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Work experience extraction failed: {e}")
            raise WorkExperienceExtractionError(f"Failed to extract work experience: {e}")
    
    async def _perform_llm_extraction(self, text: str) -> List[WorkExperience]:
        """Perform the actual LLM-based work experience extraction."""
        
        # Use function calling for reliable extraction
        structured_llm = self.openai_service.llm.with_structured_output(
            WorkExperienceLLMResult,
            method="function_calling"
        )
        
        # Create specialized work experience extraction prompt
        system_prompt = """You are an expert at extracting work experience from resumes.
        Extract ALL employment history, including full-time, part-time, internships, and consulting work.

        CRITICAL: Extract dates EXACTLY as they appear in the resume. Do not add months where none exist.

        EXTRACTION GUIDELINES:
        1. Extract each job/position as a separate work experience entry
        2. Include ALL types of work: full-time, part-time, internships, co-ops, consulting, freelance
        3. For missing information, use "Not specified" rather than making assumptions
        4. Clean company names (remove Inc., LLC, Corp. suffixes if appropriate)
        5. Standardize job titles for consistency
        6. Extract specific achievements with quantifiable results when mentioned
        7. Identify all technologies, tools, and skills used in each role

        WHAT TO EXTRACT FOR EACH POSITION:
        - Job title/position name (clean format)
        - Company name (standardized)
        - Work location (city, state/country if mentioned)
        - Start date (EXACT format from resume - do not modify)
        - End date (EXACT format from resume - use "Present" for current jobs)
        - Comprehensive job description combining all responsibilities
        - Key achievements with specific metrics/results when available
        - Technologies, tools, programming languages, and frameworks used

        DATE EXTRACTION RULES - CRITICAL:
        - If resume shows "2020 - Present" → start_date: "2020", end_date: "Present"
        - If resume shows "2018 - 2020" → start_date: "2018", end_date: "2020"  
        - If resume shows "Jan 2020 - Jun 2022" → start_date: "01/2020", end_date: "06/2022"
        - If resume shows "01/2020 - 06/2022" → start_date: "01/2020", end_date: "06/2022"
        - DO NOT add months to year-only dates
        - DO NOT convert between date formats
        - Preserve the exact format shown in the original text

        CONTENT EXTRACTION RULES:
        - Combine job description bullets into a flowing, comprehensive summary
        - Extract quantified achievements (percentages, dollar amounts, user counts, etc.)
        - List ALL technologies mentioned in connection with each role
        - Include team size, project scope, and leadership responsibilities
        - Note any promotions or role changes within the same company as separate entries
        - Focus on impact and results, not just responsibilities

        COMPANY NAME STANDARDIZATION:
        - Remove unnecessary suffixes: "TechCorp Inc." → "TechCorp"
        - Keep well-known abbreviations: "IBM Corp" → "IBM"
        - Preserve unique company names exactly as written
        - Clean up spacing and formatting inconsistencies"""

        human_prompt = """Extract all work experience from this resume text:

        RESUME TEXT:
        {text}

        Look for work experience in sections like:
        - "Work Experience" / "Professional Experience" / "Employment History"
        - "Experience" / "Career History" / "Professional Background"
        - Any chronological listing of jobs and positions

        For each position found, extract:
        1. **Complete Job Information**: Title, company, location, dates
        2. **Detailed Responsibilities**: What they did in the role
        3. **Specific Achievements**: Quantified results and accomplishments  
        4. **Technical Skills**: Technologies, tools, programming languages used
        5. **Leadership/Scope**: Team size, project scale, management duties

        CRITICAL: Extract dates exactly as written in the resume. Do not modify the format.

        If any information is unclear or missing, extract what is available and use "Not specified" for missing fields.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"text": text})
        
        return result.experiences
    
    def _post_process_experiences(self, experiences: List[WorkExperience]) -> List[WorkExperience]:
        """Post-process and validate extracted work experiences."""
        
        processed = []
        
        for exp in experiences:
            try:
                # Clean and validate the experience entry
                cleaned_exp = self._clean_experience_entry(exp)
                
                # Skip entries with missing critical information
                if not cleaned_exp.title or not cleaned_exp.company:
                    logger.warning(f"Skipping incomplete experience: {cleaned_exp.title} at {cleaned_exp.company}")
                    continue
                
                # Skip very short descriptions that might be parsing errors
                if len(cleaned_exp.description.strip()) < 10:
                    logger.warning(f"Skipping experience with very short description: {cleaned_exp.title}")
                    continue
                
                processed.append(cleaned_exp)
                
            except Exception as e:
                logger.warning(f"Error processing experience entry: {e}")
                continue
        
        # Sort by date (most recent first, current positions first)
        processed.sort(
            key=lambda x: (
                x.end_date.lower() != 'present',  # Present positions first
                self._parse_date_for_sorting(x.end_date)
            ),
            reverse=True
        )
        
        return processed
    
    def _clean_experience_entry(self, exp: WorkExperience) -> WorkExperience:
        """Clean and standardize a single work experience entry."""
        
        # Clean job title
        title = exp.title.strip() if exp.title else ""
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        title = title.title() if title.islower() else title  # Fix case if all lowercase
        
        # Clean company name
        company = exp.company.strip() if exp.company else ""
        company = re.sub(r'\s+', ' ', company)  # Normalize whitespace
        # Remove common suffixes for cleaner display
        company = re.sub(r'\s+(Inc\.?|LLC\.?|Corp\.?|Corporation|Ltd\.?|Limited)\s*$', '', company, flags=re.IGNORECASE)
        
        # Clean location
        location = exp.location.strip() if exp.location else None
        if location:
            location = re.sub(r'\s+', ' ', location)
        
        # Clean dates (minimal processing - preserve format)
        start_date = self._clean_date(exp.start_date)
        end_date = self._clean_date(exp.end_date)
        
        # Clean description
        description = exp.description.strip() if exp.description else ""
        description = re.sub(r'\s+', ' ', description)  # Normalize whitespace
        description = re.sub(r'^[•\-\*]\s*', '', description)  # Remove leading bullets
        
        # Process achievements - remove duplicates and clean
        achievements = []
        if exp.key_achievements:
            seen = set()
            for achievement in exp.key_achievements:
                clean_achievement = achievement.strip()
                clean_achievement = re.sub(r'^[•\-\*]\s*', '', clean_achievement)  # Remove bullets
                if clean_achievement and clean_achievement.lower() not in seen:
                    achievements.append(clean_achievement)
                    seen.add(clean_achievement.lower())
        
        # Process technologies - remove duplicates and standardize
        technologies = []
        if exp.technologies_used:
            seen = set()
            for tech in exp.technologies_used:
                clean_tech = tech.strip()
                if clean_tech and clean_tech.lower() not in seen:
                    technologies.append(clean_tech)
                    seen.add(clean_tech.lower())
        
        return WorkExperience(
            title=title,
            company=company,
            location=location,
            start_date=start_date,
            end_date=end_date,
            description=description,
            key_achievements=achievements,
            technologies_used=technologies
        )
    
    def _clean_date(self, date_str: str) -> str:
        """Minimal date cleaning - preserve original format."""
        if not date_str:
            return "Not specified"
        
        date_str = date_str.strip()
        
        # Standardize "Present" variations
        if date_str.lower() in ['present', 'current', 'now', 'ongoing', 'today']:
            return "Present"
        
        # Return as-is for all other cases to preserve original format
        return date_str
    
    def _parse_date_for_sorting(self, date_str: str) -> tuple:
        """Parse date for sorting purposes only."""
        if date_str.lower() == 'present':
            return (9999, 12)  # Sort present positions first
        
        try:
            if '/' in date_str:
                month, year = map(int, date_str.split('/'))
                return (year, month)
            else:
                year = int(date_str)
                return (year, 12)  # Assume December for year-only dates
        except (ValueError, TypeError):
            return (0, 0)  # Sort unparseable dates last
    
    def _calculate_total_years(self, experiences: List[WorkExperience]) -> float:
        """Calculate total years of experience for metadata."""
        result = WorkExperienceExtractionResult(experiences=experiences)
        return result.get_total_years_experience()


# Global extractor instance
llm_work_experience_extractor = LLMWorkExperienceExtractor()


# Convenience functions
async def extract_work_experience_from_resume(resume_text: str) -> WorkExperienceExtractionResult:
    """Extract work experience from resume text."""
    return await llm_work_experience_extractor.extract_work_experience_from_text(resume_text)


async def extract_work_experience_list(resume_text: str) -> List[WorkExperience]:
    """Extract work experience and return just the list of WorkExperience objects."""
    result = await extract_work_experience_from_resume(resume_text)
    return result.experiences


if __name__ == "__main__":
   
    # Test with real file if available
    import asyncio
    try:
        with open("/home/user/ranalyser/tests/data/sample_resumes/senior_developer.txt", "r") as file:
            resume_text = file.read()
        
        print("=" * 60)
        print("Testing with real resume file...")
        result = asyncio.run(extract_work_experience_from_resume(resume_text))
        
        print("✅ Real file test successful!")
        print(f"Found {len(result.experiences)} positions")
        print(f"Processing time: {result.extraction_metadata['processing_time']:.2f}s")
        
        for i, exp in enumerate(result.experiences, 1):
            print(f"{i}. {exp.title} at {exp.company} ({exp.start_date} - {exp.end_date})")
            
    except FileNotFoundError:
        print("Sample resume file not found, skipping file test")
    except Exception as e:
        print(f"Real file test error: {e}")