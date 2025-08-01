"""
LLM-based personal information extraction from resume text.
Uses OpenAI/LangChain with structured outputs for reliable data extraction.
"""
import logging
import time
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from src.models.resume import PersonalInfo
from src.services.openai_service import langgraph_openai_service, OpenAIServiceError
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonalInfoExtractionError(Exception):
    """Custom exception for personal info extraction errors."""
    pass


class PersonalInfoExtractionResult(BaseModel):
    """Personal info extraction result with metadata."""
    personal_info: PersonalInfo
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMPersonalInfoExtractor:
    """LLM-based personal information extraction system."""
    
    def __init__(self):
        """Initialize the personal info extractor."""
        self.openai_service = langgraph_openai_service
        logger.info("LLM Personal Info Extractor initialized")
    
    async def extract_personal_info_from_text(self, text: str) -> PersonalInfoExtractionResult:
        """
        Extract personal information from resume text using LLM.
        
        Args:
            text: The resume text to extract personal info from
        
        Returns:
            PersonalInfoExtractionResult with extracted data and metadata
        """
        start_time = time.time()
        
        try:
            logger.info("Starting LLM personal info extraction")
            
            # Perform LLM extraction
            personal_info = await self._perform_llm_extraction(text)
            
            # Calculate metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time": processing_time,
                "model_used": settings.openai_model,
                "extraction_method": "llm_structured"
            }
            
            result = PersonalInfoExtractionResult(
                personal_info=personal_info,
                extraction_metadata=metadata
            )
            
            logger.info(f"Personal info extraction completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Personal info extraction failed: {e}")
            raise PersonalInfoExtractionError(f"Failed to extract personal info: {e}")
    
    async def _perform_llm_extraction(self, text: str) -> PersonalInfo:
        """Perform the actual LLM-based personal info extraction."""
        
        # Create structured LLM that returns PersonalInfo directly
        structured_llm = self.openai_service.llm.with_structured_output(PersonalInfo, method="function_calling")
        
        # Create extraction prompt
        system_prompt = """You are an expert at extracting personal information from resumes. 
        Extract the contact and personal details accurately from the resume text.

EXTRACTION GUIDELINES:
1. Extract only information that is explicitly mentioned in the text
2. For missing information, use None/null values
3. Clean and standardize the extracted data
4. Ensure URLs are complete and properly formatted
5. Extract the full name as it appears (don't split into first/last)

WHAT TO EXTRACT:
- Full name of the person
- Email address (if present)
- Phone number (if present) 
- Location/address (city, state, country as mentioned)
- LinkedIn profile URL (if present)
- Portfolio/website URL (if present)
- GitHub profile URL (if present)

FORMATTING RULES:
- Keep phone numbers in their original format
- Ensure URLs include http:// or https:// if missing
- For location, extract as mentioned (could be "City, State" or "City, Country", etc.)
- Use exact name capitalization as shown in resume"""

        human_prompt = """Extract personal information from this resume text:

RESUME TEXT:
{text}

Focus on finding:
1. The person's full name (usually at the top)
2. Contact information (email, phone)
3. Location/address information
4. Social media and portfolio links (LinkedIn, GitHub, personal website)

Extract only what is clearly present in the text. If something is not mentioned, leave it as null."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        
        chain = prompt | structured_llm
        
        result = await chain.ainvoke({"text": text})
        
        return result


# Global extractor instance
llm_personal_info_extractor = LLMPersonalInfoExtractor()


# Convenience functions
async def extract_personal_info_from_resume(resume_text: str) -> PersonalInfoExtractionResult:
    """Extract personal info from resume text."""
    return await llm_personal_info_extractor.extract_personal_info_from_text(resume_text)


async def extract_personal_info_quick(resume_text: str) -> PersonalInfo:
    """Extract personal info and return just the PersonalInfo object."""
    result = await extract_personal_info_from_resume(resume_text)
    return result.personal_info


if __name__ == "__main__":
    # Example usage
    with open("/home/user/ranalyser/tests/data/sample_resumes/senior_developer.txt", "r") as file:
        resume_text = file.read()
    import asyncio
    result = asyncio.run(extract_personal_info_from_resume(resume_text))
    print(result)