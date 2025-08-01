"""
LLM-based education extraction from resume text.
Uses OpenAI/LangChain with function calling for reliable extraction of educational background.
Preserves original date formats and handles various education entry types.
"""
import logging
import time
import re
from typing import Dict, List, Optional, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from src.models.resume import Education
from src.services.openai_service import langgraph_openai_service, OpenAIServiceError
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EducationExtractionError(Exception):
    """Custom exception for education extraction errors."""
    pass


class EducationExtractionResult(BaseModel):
    """Education extraction result with metadata."""
    education: List[Education] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_highest_degree(self) -> Optional[Education]:
        """Get the highest level degree."""
        if not self.education:
            return None
        
        # Define degree hierarchy
        degree_hierarchy = {
            'phd': 6, 'doctorate': 6, 'doctoral': 6, 'ph.d': 6,
            'master': 5, 'masters': 5, 'mba': 5, 'ms': 5, 'ma': 5, 'm.s': 5, 'm.a': 5,
            'bachelor': 4, 'bachelors': 4, 'bs': 4, 'ba': 4, 'btech': 4, 'b.tech': 4, 'b.s': 4, 'b.a': 4,
            'associate': 3, 'associates': 3, 'aa': 3, 'as': 3,
            'diploma': 2, 'certificate': 1, 'certification': 1
        }
        
        highest_level = 0
        highest_education = None
        
        for edu in self.education:
            degree_lower = edu.degree.lower()
            for key, level in degree_hierarchy.items():
                if key in degree_lower and level > highest_level:
                    highest_level = level
                    highest_education = edu
                    break
        
        return highest_education or self.education[0]
    
    def get_recent_education(self) -> Optional[Education]:
        """Get the most recent education entry."""
        if not self.education:
            return None
        
        # Sort by graduation date (most recent first)
        sorted_education = sorted(
            [edu for edu in self.education if edu.graduation_date],
            key=lambda x: self._parse_date_for_sorting(x.graduation_date),
            reverse=True
        )
        
        return sorted_education[0] if sorted_education else self.education[0]
    
    def get_institutions_attended(self) -> List[str]:
        """Get list of unique institutions attended."""
        return list(set(edu.institution for edu in self.education))
    
    def has_degree_in_field(self, field: str) -> bool:
        """Check if candidate has a degree in a specific field."""
        field_lower = field.lower()
        for edu in self.education:
            if field_lower in edu.field.lower() or field_lower in edu.degree.lower():
                return True
        return False
    
    def _parse_date_for_sorting(self, date_str: str) -> int:
        """Parse date for sorting purposes."""
        try:
            if '/' in date_str:
                parts = date_str.split('/')
                return int(parts[-1])  # Get year part
            else:
                return int(date_str)
        except (ValueError, TypeError):
            return 0


class EducationLLMResult(BaseModel):
    """Wrapper for LLM extraction result."""
    education: List[Education] = Field(default_factory=list)


class LLMEducationExtractor:
    """LLM-based education extraction system."""
    
    def __init__(self):
        """Initialize the education extractor."""
        self.openai_service = langgraph_openai_service
        logger.info("LLM Education Extractor initialized")
    
    async def extract_education_from_text(self, text: str) -> EducationExtractionResult:
        """
        Extract education from resume text using LLM.
        
        Args:
            text: The resume text to extract education from
        
        Returns:
            EducationExtractionResult with extracted education data and metadata
        """
        start_time = time.time()
        
        try:
            logger.info("Starting LLM education extraction")
            
            # Perform LLM extraction
            education_list = await self._perform_llm_extraction(text)
            
            # Post-process education entries
            processed_education = self._post_process_education(education_list)
            
            # Calculate metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time": processing_time,
                "total_education_entries": len(processed_education),
                "institutions": list(set(edu.institution for edu in processed_education)),
                "highest_degree": self._get_highest_degree_name(processed_education),
                "model_used": settings.openai_model,
                "extraction_method": "llm_function_calling"
            }
            
            result = EducationExtractionResult(
                education=processed_education,
                extraction_metadata=metadata
            )
            
            logger.info(f"Education extraction completed. Found {len(processed_education)} entries in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Education extraction failed: {e}")
            raise EducationExtractionError(f"Failed to extract education: {e}")
    
    async def _perform_llm_extraction(self, text: str) -> List[Education]:
        """Perform the actual LLM-based education extraction."""
        
        # Use function calling for reliable extraction
        structured_llm = self.openai_service.llm.with_structured_output(
            EducationLLMResult,
            method="function_calling"
        )
        
        # Create specialized education extraction prompt
        system_prompt = """You are an expert at extracting formal educational background from resumes.
        Extract ONLY formal education entries - degrees and academic programs from educational institutions.

        DO NOT EXTRACT:
        - Professional certifications (AWS, Google, Microsoft, etc.)
        - Industry certifications (PMP, CISSP, etc.)
        - Professional licenses
        - Training programs or workshops
        - Corporate training or seminars

        EXTRACT ONLY:
        - Formal degrees (Bachelor's, Master's, PhD, Associate's)
        - Academic diplomas from schools/universities
        - Formal academic programs and courses from educational institutions
        - Bootcamps and intensive academic programs (coding bootcamps, etc.)

        EXTRACTION GUIDELINES:
        1. Extract each formal degree/education as a separate entry
        2. Focus on degrees from universities, colleges, schools, and academic institutions
        3. Extract graduation dates exactly as shown in the resume
        4. For missing information, use "Not specified" rather than guessing
        5. Standardize degree names and institution names appropriately
        6. Extract GPA, honors, and awards when mentioned
        7. Include relevant coursework or specializations when noted

        WHAT TO EXTRACT FOR EACH EDUCATION ENTRY:
        - Degree type and level (Bachelor's, Master's, PhD, Certificate, etc.)
        - Field of study/major (Computer Science, Business, etc.)
        - Institution name (university, college, school)
        - Graduation date (preserve original format)
        - GPA (if mentioned)
        - Honors/awards (Dean's List, Magna Cum Laude, etc.)

        DEGREE STANDARDIZATION:
        - "BS" or "B.S." → "Bachelor of Science"
        - "BA" or "B.A." → "Bachelor of Arts"
        - "MS" or "M.S." → "Master of Science"
        - "MA" or "M.A." → "Master of Arts"
        - "MBA" → "Master of Business Administration"
        - "PhD" or "Ph.D." → "Doctor of Philosophy"
        - Keep original if unclear or non-standard

        INSTITUTION STANDARDIZATION:
        - Use full, official names when recognizable
        - "UC Berkeley" → "University of California, Berkeley"
        - "MIT" → "Massachusetts Institute of Technology"
        - Keep original name if uncertain

        DATE EXTRACTION RULES:
        - Extract graduation dates exactly as shown
        - Common formats: "2020", "May 2020", "05/2020", "2016-2020"
        - Use "Expected [date]" for future graduations
        - Use "Not specified" if no date is mentioned

        FIELD OF STUDY RULES:
        - Extract the specific major/field mentioned
        - "Computer Science", "Electrical Engineering", "Business Administration"
        - Include concentrations: "Computer Science with focus on AI"
        - Use "Not specified" if field is unclear

        GPA AND HONORS EXTRACTION:
        - Extract GPA exactly as mentioned: "3.8", "3.85/4.0"
        - Common honors: "Summa Cum Laude", "Magna Cum Laude", "Cum Laude"
        - Other recognitions: "Dean's List", "Honor Roll", "Phi Beta Kappa"
        - Academic awards and scholarships"""

        human_prompt = """Extract ONLY formal educational background from this resume text:

        RESUME TEXT:
        {text}

        Look for formal education in sections like:
        - "Education" / "Educational Background" / "Academic Background"
        - "Academic Qualifications" / "Degrees"
        - Any section listing degrees from universities, colleges, or schools

        IMPORTANT: Extract ONLY formal education from academic institutions:
        ✅ INCLUDE:
        - University degrees (Bachelor's, Master's, PhD)
        - College diplomas and associate degrees
        - Academic programs from schools and universities
        - Formal academic courses and bootcamps
        - Study abroad programs

        ❌ DO NOT INCLUDE:
        - Professional certifications (AWS, Microsoft, Google, etc.)
        - Industry certifications (PMP, CISSP, Salesforce, etc.)
        - Professional licenses (CPA, PE, etc.)
        - Corporate training programs
        - Professional workshops or seminars

        For each FORMAL EDUCATION entry found, extract:
        1. **Degree Information**: Type of degree and field of study
        2. **Institution Details**: School/university name and location if mentioned
        3. **Timeline**: Graduation date or time period
        4. **Academic Performance**: GPA, honors, awards if mentioned
        5. **Additional Details**: Relevant coursework, thesis topics, specializations

        Extract dates exactly as written in the resume. Do not modify the format.

        If any information is missing or unclear, use "Not specified" for that field.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"text": text})
        
        return result.education
    
    def _post_process_education(self, education_list: List[Education]) -> List[Education]:
        """Post-process and validate extracted education entries."""
        
        processed = []
        
        for edu in education_list:
            try:
                # Clean and validate the education entry
                cleaned_edu = self._clean_education_entry(edu)
                
                # Skip entries with missing critical information
                if not cleaned_edu.degree or not cleaned_edu.institution:
                    logger.warning(f"Skipping incomplete education: {cleaned_edu.degree} at {cleaned_edu.institution}")
                    continue
                
                # Skip very generic or unclear entries
                if len(cleaned_edu.degree.strip()) < 3 or len(cleaned_edu.institution.strip()) < 3:
                    logger.warning(f"Skipping unclear education entry: {cleaned_edu.degree}")
                    continue
                
                processed.append(cleaned_edu)
                
            except Exception as e:
                logger.warning(f"Error processing education entry: {e}")
                continue
        
        # Sort by graduation date (most recent first)
        processed.sort(
            key=lambda x: self._parse_graduation_date_for_sorting(x.graduation_date),
            reverse=True
        )
        
        return processed
    
    def _clean_education_entry(self, edu: Education) -> Education:
        """Clean and standardize a single education entry."""
        
        # Clean degree name
        degree = edu.degree.strip() if edu.degree else ""
        degree = re.sub(r'\s+', ' ', degree)  # Normalize whitespace
        degree = self._standardize_degree_name(degree)
        
        # Clean field of study
        field = edu.field.strip() if edu.field else ""
        field = re.sub(r'\s+', ' ', field)  # Normalize whitespace
        
        # Clean institution name
        institution = edu.institution.strip() if edu.institution else ""
        institution = re.sub(r'\s+', ' ', institution)  # Normalize whitespace
        #institution = self._standardize_institution_name(institution)
        
        # Clean graduation date (preserve format)
        graduation_date = self._clean_graduation_date(edu.graduation_date)
        
        # Clean GPA
        gpa = edu.gpa.strip() if edu.gpa else None
        if gpa and gpa.lower() in ['not specified', 'n/a', 'na', 'none']:
            gpa = None
        
        # Clean honors
        honors = edu.honors.strip() if edu.honors else None
        if honors and honors.lower() in ['not specified', 'n/a', 'na', 'none']:
            honors = None
        
        return Education(
            degree=degree,
            field=field,
            institution=institution,
            graduation_date=graduation_date,
            gpa=gpa,
            honors=honors
        )
    
    def _standardize_degree_name(self, degree: str) -> str:
        """Standardize degree names for consistency."""
        if not degree:
            return degree
        
        degree_lower = degree.lower()
        
        # Common degree standardizations
        standardizations = {
            'bs': 'Bachelor of Science',
            'b.s': 'Bachelor of Science',
            'b.s.': 'Bachelor of Science',
            'bachelor of science': 'Bachelor of Science',
            'ba': 'Bachelor of Arts',
            'b.a': 'Bachelor of Arts',
            'b.a.': 'Bachelor of Arts',
            'bachelor of arts': 'Bachelor of Arts',
            'btech': 'Bachelor of Technology',
            'b.tech': 'Bachelor of Technology',
            'be': 'Bachelor of Engineering',
            'b.e': 'Bachelor of Engineering',
            'ms': 'Master of Science',
            'm.s': 'Master of Science',
            'm.s.': 'Master of Science',
            'master of science': 'Master of Science',
            'ma': 'Master of Arts',
            'm.a': 'Master of Arts',
            'm.a.': 'Master of Arts',
            'master of arts': 'Master of Arts',
            'mba': 'Master of Business Administration',
            'm.b.a': 'Master of Business Administration',
            'mtech': 'Master of Technology',
            'm.tech': 'Master of Technology',
            'phd': 'Doctor of Philosophy',
            'ph.d': 'Doctor of Philosophy',
            'ph.d.': 'Doctor of Philosophy',
            'doctorate': 'Doctor of Philosophy'
        }
        
        for key, standard in standardizations.items():
            if degree_lower == key or degree_lower == key + '.':
                return standard
        
        # If no standardization found, return title case
        return degree.title()
    
    def _standardize_institution_name(self, institution: str) -> str:
        """Standardize institution names for consistency."""
        if not institution:
            return institution
        
        # Common institution standardizations
        institution_lower = institution.lower()
        
        standardizations = {
            'mit': 'Massachusetts Institute of Technology',
            'stanford': 'Stanford University',
            'harvard': 'Harvard University',
            'uc berkeley': 'University of California, Berkeley',
            'berkeley': 'University of California, Berkeley',
            'caltech': 'California Institute of Technology',
            'carnegie mellon': 'Carnegie Mellon University',
            'cmu': 'Carnegie Mellon University',
            'georgia tech': 'Georgia Institute of Technology',
            'gt': 'Georgia Institute of Technology'
        }
        
        for key, standard in standardizations.items():
            if key in institution_lower:
                return standard
        
        # Return original with proper title case
        return institution.title()
    
    def _clean_graduation_date(self, date_str: Optional[str]) -> Optional[str]:
        """Clean graduation date while preserving format."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Handle common variations
        if date_str.lower() in ['not specified', 'n/a', 'na', 'none', 'tbd', 'pending']:
            return None
        
        # Handle "Expected" dates
        if 'expected' in date_str.lower():
            return date_str
        
        return date_str
    
    def _parse_graduation_date_for_sorting(self, date_str: Optional[str]) -> int:
        """Parse graduation date for sorting purposes."""
        if not date_str:
            return 0
        
        # Extract year from various formats
        try:
            # Look for 4-digit year
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                return int(year_match.group())
            
            # Try direct conversion
            return int(date_str)
        except (ValueError, TypeError):
            return 0
    
    def _get_highest_degree_name(self, education_list: List[Education]) -> str:
        """Get the name of the highest degree for metadata."""
        if not education_list:
            return "None"
        
        result = EducationExtractionResult(education=education_list)
        highest = result.get_highest_degree()
        return highest.degree if highest else "Not specified"


# Global extractor instance
llm_education_extractor = LLMEducationExtractor()


# Convenience functions
async def extract_education_from_resume(resume_text: str) -> EducationExtractionResult:
    """Extract education from resume text."""
    return await llm_education_extractor.extract_education_from_text(resume_text)


async def extract_education_list(resume_text: str) -> List[Education]:
    """Extract education and return just the list of Education objects."""
    result = await extract_education_from_resume(resume_text)
    return result.education


if __name__ == "__main__":
   
    try:
        with open("/home/user/ranalyser/tests/data/sample_resumes/senior_developer.txt", "r") as file:
            resume_text = file.read()
        import asyncio
        print("=" * 60)
        print("Testing with real resume file...")
        result = asyncio.run(extract_education_from_resume(resume_text))
        
        print("✅ Real file test successful!")
        print(f"Found {len(result.education)} education entries")
        print(f"Processing time: {result.extraction_metadata['processing_time']:.2f}s")
        
        for i, edu in enumerate(result.education, 1):
            print(f"{i}. {edu.degree} in {edu.field}")
            print(f"   {edu.institution} ({edu.graduation_date})")
            
    except FileNotFoundError:
        print("Sample resume file not found, skipping file test")
    except Exception as e:
        print(f"Real file test error: {e}")