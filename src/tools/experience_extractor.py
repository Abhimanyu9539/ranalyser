"""
LLM-based work experience extraction from resume text.
Uses OpenAI/LangChain with function calling for reliable extraction of work history.
Natural date handling - preserves original format (year-only vs month/year).
"""
import logging
import time
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel, Field

from src.models.resume import WorkExperience
from src.services.openai_service import langgraph_openai_service, OpenAIServiceError
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperienceExtractionError(Exception):
    """Custom exception for experience extraction errors."""
    pass


class ExperienceExtractionResult(BaseModel):
    """Work experience extraction result with metadata"""
    experiences: List[WorkExperience] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_total_years_experience(self) -> float:
        """Calculate total years of experience across all positions"""
        total_years = 0.0
        current_year = datetime.now().year
        current_month = datetime.now().month

        for exp in self.experiences:
            try:
                # Check if both dates have months or both are year-only
                start_has_month = '/' in exp.start_date
                end_has_month = '/' in exp.end_date and exp.end_date.lower() != 'present'
                is_present = exp.end_date.lower() == 'present'
                
                if start_has_month or end_has_month or is_present:
                    # At least one date has month info, or it's current - use month-level calculation
                    if start_has_month:
                        start_parts = exp.start_date.split('/')
                        start_month, start_year = int(start_parts[0]), int(start_parts[1])
                    else:
                        start_year = int(exp.start_date)
                        start_month = 1  # Assume January for year-only start
                    
                    if is_present:
                        end_month, end_year = current_month, current_year
                    elif end_has_month:
                        end_parts = exp.end_date.split('/')
                        end_month, end_year = int(end_parts[0]), int(end_parts[1])
                    else:
                        end_year = int(exp.end_date)
                        end_month = 12  # Assume December for year-only end
                    
                    months = ((end_year - start_year) * 12) + (end_month - start_month)
                    total_years += max(0, months / 12)
                else:
                    # Both dates are year-only - use simple year subtraction
                    start_year = int(exp.start_date)
                    end_year = int(exp.end_date)
                    years = end_year - start_year + 1  # +1 because if someone worked 2020-2020, that's 1 year
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
        """Get list of unique companies"""
        return list(set(exp.company for exp in self.experiences))


class ExperienceExtractionLLMResult(BaseModel):
    """Wrapper for list of work experiences - this is what the LLM returns"""
    experiences: List[WorkExperience] = Field(default_factory=list)


class LLMExperienceExtractor:
    """LLM-based work experience extraction system"""
    
    def __init__(self):
        """Initialize the experience extractor"""
        self.openai_service = langgraph_openai_service
        logger.info("LLM Experience Extractor Initialized")

    async def extract_experience_from_text(self, text: str) -> ExperienceExtractionResult:
        """
        Extract work experience from resume text using LLM.
        
        Args:
            text: The resume text to extract experience from
        
        Returns:
            ExperienceExtractionResult with extracted experience data and metadata
        """
        start_time = time.time()

        try: 
            logger.info("Starting LLM work experience extraction")

            # Perform LLM Extraction
            experiences = await self._perform_llm_extraction(text)

            # Post process experiences
            processed_experiences = self._post_process_experiences(experiences)

            # Calculate metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time": processing_time,
                "total_positions": len(processed_experiences),
                "current_positions": len([exp for exp in processed_experiences if exp.end_date.lower() == 'present']),
                "companies": list(set(exp.company for exp in processed_experiences)),
                "model_used": settings.openai_model,
                "extraction_method": "llm_function_calling"
            }

            result = ExperienceExtractionResult(
                experiences=processed_experiences,
                extraction_metadata=metadata
            )
            
            logger.info(f"Experience extraction completed. Found {len(processed_experiences)} positions in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Experience extraction failed: {e}")
            raise ExperienceExtractionError(f"Failed to extract work experience: {e}")
    
    async def _perform_llm_extraction(self, text: str) -> List[WorkExperience]:
        """Perform LLM-based experience extraction"""

        # Use the wrapper model for structured output
        structured_llm = self.openai_service.llm.with_structured_output(
            ExperienceExtractionLLMResult,
            method="function_calling"
        )

        # Create extraction prompt
        system_prompt = """You are an expert at extracting work experience information from resumes.
        Extract all work positions, jobs, and employment history accurately from the resume text.

        EXTRACTION GUIDELINES:
        1. Extract each position as a separate work experience entry
        2. Include internships, co-ops, part-time, and full-time positions
        3. For missing information, use reasonable defaults or "Not specified"
        4. Standardize company names (remove "Inc.", "LLC", etc. if needed for clarity)
        5. Clean and format job titles consistently
        6. Extract quantified achievements and key accomplishments
        7. Identify technologies, tools, and skills mentioned in job descriptions

        WHAT TO EXTRACT FOR EACH POSITION:
        - Job title/position name
        - Company name (clean format)
        - Location (city, state/country if mentioned)
        - Start date and end date (preserve original format)
        - Job description/summary of role and responsibilities
        - Key achievements (specific accomplishments, metrics, impacts)
        - Technologies used (programming languages, tools, frameworks, etc.)

        DATE FORMATTING RULES:
        - Extract dates EXACTLY as they appear in the resume
        - If resume shows "2020 - Present", extract start_date: "2020", end_date: "Present"
        - If resume shows "Jan 2020 - Jun 2022", extract start_date: "01/2020", end_date: "06/2022"
        - If resume shows "01/2020 - 06/2022", extract start_date: "01/2020", end_date: "06/2022"
        - DO NOT add months if only years are shown in the resume
        - DO NOT convert year-only dates to MM/YYYY format

        CONTENT EXTRACTION RULES:
        - Combine role description and bullet points into a comprehensive description
        - Extract specific achievements with numbers/metrics when mentioned
        - List technologies, programming languages, tools used in each role
        - Look for leadership responsibilities, team size, project scope
        - Include any promotions or role changes within the same company as separate entries

        QUALITY CHECKS:
        - Ensure all positions have required fields (title, company, dates)
        - Verify date consistency (start date before end date)
        - Clean up formatting and remove extra whitespace
        - Standardize similar job titles and company names"""

        human_prompt = """Extract all work experience and employment history from this resume text:

        RESUME TEXT:
        {text}

        Focus on finding:
        1. **Job Positions**: All employment, internships, co-ops, consulting work
        2. **Company Information**: Names and locations of employers
        3. **Employment Dates**: Start and end dates for each position (preserve original format)
        4. **Role Details**: Job responsibilities, achievements, and technologies used
        5. **Career Progression**: Promotions, role changes, career growth

        Extract each position as a separate entry, even if at the same company.
        Look for experience in sections like "Work Experience", "Employment History", "Professional Experience", etc.

        IMPORTANT: For dates, extract EXACTLY as shown in the resume:
        - If resume shows "2020 - Present", extract as start_date: "2020", end_date: "Present"
        - If resume shows "2018 - 2020", extract as start_date: "2018", end_date: "2020"
        - If resume shows "Jan 2020 - Jun 2022", extract as start_date: "01/2020", end_date: "06/2022"
        - If resume shows "01/2020 - 06/2022", extract as start_date: "01/2020", end_date: "06/2022"
        - DO NOT add months to year-only dates
        - DO NOT convert formats - preserve exactly as written

        If a position lacks certain details, extract what's available and note missing information appropriately.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt), 
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])

        chain = prompt | structured_llm
        result = await chain.ainvoke({'text': text})
        return result.experiences

    def _post_process_experiences(self, experiences: List[WorkExperience]) -> List[WorkExperience]:
        """Post-process extracted experiences for consistency and validation."""
        
        processed = []
        
        for exp in experiences:
            try:
                # Clean and validate the experience entry
                cleaned_exp = self._clean_experience_entry(exp)
                
                logger.debug(f"Cleaned experience: {cleaned_exp.title} at {cleaned_exp.company}")
                
                # Skip if missing critical information
                if not cleaned_exp.title or not cleaned_exp.company:
                    logger.warning(f"Skipping experience entry with missing critical info: {cleaned_exp.title} at {cleaned_exp.company}")
                    continue
                
                processed.append(cleaned_exp)
                
            except Exception as e:
                logger.warning(f"Error processing experience entry: {e}")
                continue
        
        # Sort by end date (most recent first, with "Present" positions first)
        processed.sort(
            key=lambda x: (
                x.end_date.lower() != 'present',  # Present positions first
                self._parse_date_for_sorting(x.end_date)
            ),
            reverse=True
        )
        
        return processed
    
    def _clean_experience_entry(self, exp: WorkExperience) -> WorkExperience:
        """Clean and standardize a single experience entry."""
        
        # Clean title
        title = exp.title.strip() if exp.title else ""
        if title:
            title = re.sub(r'\s+', ' ', title)  # Remove extra whitespace
        
        # Clean company name
        company = exp.company.strip() if exp.company else ""
        if company:
            # Remove common corporate suffixes for cleaner display
            company = re.sub(r'\s+(Inc\.?|LLC\.?|Corp\.?|Corporation|Ltd\.?|Limited)\s*$', '', company, flags=re.IGNORECASE)
            company = re.sub(r'\s+', ' ', company)
        
        # Clean location
        location = exp.location.strip() if exp.location else None
        if location:
            location = re.sub(r'\s+', ' ', location)
        
        # Standardize dates (preserving original format)
        start_date = self._standardize_date(exp.start_date)
        end_date = self._standardize_date(exp.end_date)
        
        # Clean description
        description = exp.description.strip() if exp.description else ""
        if description:
            description = re.sub(r'\s+', ' ', description)
        
        # Clean and deduplicate achievements
        achievements = []
        if exp.key_achievements:
            seen_achievements = set()
            for achievement in exp.key_achievements:
                clean_achievement = achievement.strip()
                if clean_achievement and clean_achievement not in seen_achievements:
                    achievements.append(clean_achievement)
                    seen_achievements.add(clean_achievement)
        
        # Clean and deduplicate technologies
        technologies = []
        if exp.technologies_used:
            seen_techs = set()
            for tech in exp.technologies_used:
                clean_tech = tech.strip()
                if clean_tech and clean_tech.lower() not in seen_techs:
                    technologies.append(clean_tech)
                    seen_techs.add(clean_tech.lower())
        
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
    
    def _standardize_date(self, date_str: str) -> str:
        """Standardize date format - keep original format if only year, MM/YYYY if month included."""
        if not date_str:
            return "Not specified"
        
        date_str = date_str.strip()
        
        # Handle "Present" and variations
        if date_str.lower() in ['present', 'current', 'now', 'ongoing']:
            return "Present"
        
        # Try to parse various date formats
        date_patterns = [
            # Patterns with month and year - standardize to MM/YYYY
            (r'^(\d{1,2})/(\d{4})$', 'month_year'),      # MM/YYYY or M/YYYY
            (r'^(\d{4})/(\d{1,2})$', 'year_month'),      # YYYY/MM or YYYY/M  
            (r'^(\d{1,2})-(\d{4})$', 'month_year'),      # MM-YYYY or M-YYYY
            (r'^(\d{4})-(\d{1,2})$', 'year_month'),      # YYYY-MM
            # Pattern with only year - keep as year only
            (r'^(\d{4})$', 'year_only'),                 # YYYY -> keep as YYYY
        ]
        
        for pattern, format_type in date_patterns:
            match = re.match(pattern, date_str)
            if match:
                if format_type == 'month_year':
                    # MM/YYYY or M/YYYY format - standardize month padding
                    month, year = match.groups()
                    return f"{int(month):02d}/{year}"
                elif format_type == 'year_month':
                    # YYYY/MM or YYYY-MM format - swap and standardize
                    year, month = match.groups()
                    return f"{int(month):02d}/{year}"
                elif format_type == 'year_only':
                    # YYYY format - keep as is
                    year = match.group(1)
                    return year
        
        # If we can't parse it, return as-is
        return date_str
    
    def _parse_date_for_sorting(self, date_str: str) -> tuple:
        """Parse date string for sorting purposes."""
        if date_str.lower() == 'present':
            return (9999, 12)  # Sort present positions first
        
        try:
            if '/' in date_str:
                month, year = map(int, date_str.split('/'))
                return (year, month)
            else:
                year = int(date_str)
                return (year, 12)  # Assume December for year-only dates when sorting
        except (ValueError, TypeError):
            return (0, 0)  # Sort unparseable dates last


# Global extractor instance
llm_experience_extractor = LLMExperienceExtractor()


# Convenience functions
async def extract_experience_from_resume(resume_text: str) -> ExperienceExtractionResult:
    """Extract work experience from resume text."""
    return await llm_experience_extractor.extract_experience_from_text(resume_text)


async def extract_experience_list(resume_text: str) -> List[WorkExperience]:
    """Extract work experience and return just the list of WorkExperience objects."""
    result = await extract_experience_from_resume(resume_text)
    return result.experiences


if __name__ == "__main__":
    import asyncio
    
    # Test with real file if available
    try:
        with open("/home/user/ranalyser/tests/data/sample_resumes/senior_developer.txt", "r") as file:
            resume_text = file.read()
        
        print("=" * 60)
        print("Testing with real resume file...")
        result = asyncio.run(extract_experience_from_resume(resume_text))
        print("✅ Success!")
        print(f"Found {len(result.experiences)} positions")
        print(f"Processing time: {result.extraction_metadata['processing_time']:.2f}s")
        
        for i, exp in enumerate(result.experiences, 1):
            print(f"{i}. {exp.title} at {exp.company}")
            print(f"   Dates: {exp.start_date} - {exp.end_date}")
            
    except FileNotFoundError:
        print("Sample resume file not found, skipping file test")
    except Exception as e:
        print(f"File test error: {e}")