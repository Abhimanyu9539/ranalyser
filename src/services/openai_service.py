"""
OpenAI API integration service for the Resume Analyzer application.
Updated for LangGraph integration with structured outputs using Pydantic models.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Type, TypeVar
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel

from config.settings import settings
from config.prompts import ResumePrompts, JobMatchingPrompts, ATSPrompts, ImprovementPrompts

# Import our Pydantic models
from src.models.resume import Resume, ResumeAnalysisResult, Skills
from src.models.job import Job, JobMatch, JobSearchResult
from src.models.ats_score import ATSScore, ImprovementSuggestion

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)


class OpenAIServiceError(Exception):
    """Custom exception for OpenAI service errors."""
    pass


class UsageTracker:
    """Track API usage without cost analysis."""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
        self.start_time = datetime.now()
        self.processing_times = []
    
    def add_usage(self, prompt_tokens: int, completion_tokens: int, processing_time: float = 0):
        """Add token usage from an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.call_count += 1
        if processing_time > 0:
            self.processing_times.append(processing_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "api_calls": self.call_count,
            "duration_seconds": round(duration, 2),
            "tokens_per_second": round(self.total_tokens / max(duration, 1), 2),
            "avg_processing_time": round(avg_processing_time, 2)
        }


class LangGraphOpenAIService:
    """OpenAI service optimized for LangGraph workflows with structured outputs."""
    
    def __init__(self):
        """Initialize OpenAI service with LangChain configuration."""
        self.api_key = settings.openai_api_key
        print(self.api_key)
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        
        # Initialize LangChain LLM with structured output support
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            max_retries=3,
            request_timeout=120
        )
        
        # Usage tracking
        self.usage_tracker = UsageTracker()
        
        logger.info(f"LangGraph OpenAI Service initialized with model: {self.model}")
    
    def _create_structured_llm(self, pydantic_model: Type[T]) -> ChatOpenAI:
        """Create LLM with structured output for given Pydantic model."""
        return self.llm.with_structured_output(pydantic_model)
    
    def _create_prompt_template(self, system_message: str, human_message: str) -> ChatPromptTemplate:
        """Create a chat prompt template."""
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_message)
        ])
    
    async def parse_resume_structured(self, resume_text: str) -> ResumeAnalysisResult:
        """Parse resume text into structured Resume model."""
        start_time = time.time()
        
        try:
            # Create structured LLM for Resume model
            structured_llm = self._create_structured_llm(Resume)
            
            # Create prompt using original variable name
            prompt = self._create_prompt_template(
                system_message="You are an expert resume parser. Extract structured information accurately from the resume text.",
                human_message=ResumePrompts.PARSE_RESUME
            )
            
            # Create chain
            chain = prompt | structured_llm
            
            # Invoke chain with correct variable name
            result = await chain.ainvoke({"resume_text": resume_text})
            
            processing_time = time.time() - start_time
            
            # Create analysis result
            analysis_result = ResumeAnalysisResult(
                resume=result,
                parsing_confidence=0.95,  # High confidence with structured output
                extracted_sections=self._get_extracted_sections(result),
                processing_time=processing_time,
                model_used=self.model
            )
            
            # Track usage (tokens would need to be extracted from callback if needed)
            self.usage_tracker.add_usage(0, 0, processing_time)
            
            logger.info(f"Resume parsed successfully in {processing_time:.2f} seconds")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Resume parsing failed: {e}")
            raise OpenAIServiceError(f"Resume parsing failed: {e}")
    
    def parse_resume_structured_sync(self, resume_text: str) -> ResumeAnalysisResult:
        """Synchronous version of structured resume parsing."""
        start_time = time.time()
        
        try:
            # Create structured LLM for Resume model
            structured_llm = self._create_structured_llm(Resume)
            
            # Create prompt using original variable name
            prompt = self._create_prompt_template(
                system_message="You are an expert resume parser. Extract structured information accurately from the resume text.",
                human_message=ResumePrompts.PARSE_RESUME
            )
            
            # Create chain
            chain = prompt | structured_llm
            
            # Invoke chain with correct variable name
            result = chain.invoke({"resume_text": resume_text})
            
            processing_time = time.time() - start_time
            
            # Create analysis result
            analysis_result = ResumeAnalysisResult(
                resume=result,
                parsing_confidence=0.95,  # High confidence with structured output
                extracted_sections=self._get_extracted_sections(result),
                processing_time=processing_time,
                model_used=self.model
            )
            
            # Track usage
            self.usage_tracker.add_usage(0, 0, processing_time)
            
            logger.info(f"Resume parsed successfully in {processing_time:.2f} seconds")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Resume parsing failed: {e}")
            raise OpenAIServiceError(f"Resume parsing failed: {e}")
    
    async def extract_skills_structured(self, resume_text: str) -> Skills:
        """Extract skills using structured output."""
        try:
            structured_llm = self._create_structured_llm(Skills)
            
            prompt = self._create_prompt_template(
                system_message="You are a skills extraction expert. Categorize skills accurately.",
                human_message=ResumePrompts.EXTRACT_SKILLS
            )
            
            chain = prompt | structured_llm
            result = await chain.ainvoke({"resume_text": resume_text})
            
            logger.info("Skills extracted successfully")
            return result
            
        except Exception as e:
            logger.error(f"Skills extraction failed: {e}")
            raise OpenAIServiceError(f"Skills extraction failed: {e}")
    
    async def analyze_job_match_structured(
        self, 
        resume: Resume, 
        job: Job
    ) -> JobMatch:
        """Analyze job match using structured output."""
        try:
            structured_llm = self._create_structured_llm(JobMatch)
            
            prompt = self._create_prompt_template(
                system_message="You are an expert career consultant analyzing job-resume compatibility.",
                human_message="""
                Analyze how well this resume matches the job requirements:
                
                Resume Data: {resume_data}
                Job Description: {job_description}
                
                Provide detailed match analysis including percentages, matched skills, missing skills, and recommendations.
                """
            )
            
            chain = prompt | structured_llm
            result = await chain.ainvoke({
                "resume_data": resume.model_dump_json(indent=2),
                "job_description": f"Title: {job.title}\nCompany: {job.company.name}\nDescription: {job.description}\nRequirements: {job.requirements.model_dump_json()}"
            })
            
            # Set the job reference in the result
            result.job = job
            
            logger.info("Job match analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Job match analysis failed: {e}")
            raise OpenAIServiceError(f"Job match analysis failed: {e}")
    
    async def calculate_ats_score_structured(
        self, 
        resume_text: str, 
        job_description: str
    ) -> ATSScore:
        """Calculate ATS score using structured output."""
        try:
            structured_llm = self._create_structured_llm(ATSScore)
            
            prompt = self._create_prompt_template(
                system_message="You are an ATS expert analyzing resume compatibility with tracking systems.",
                human_message=ATSPrompts.CALCULATE_ATS_SCORE
            )
            
            chain = prompt | structured_llm
            result = await chain.ainvoke({
                "resume_text": resume_text,
                "job_description": job_description
            })
            
            logger.info(f"ATS score calculated: {result.overall_score}/100")
            return result
            
        except Exception as e:
            logger.error(f"ATS score calculation failed: {e}")
            raise OpenAIServiceError(f"ATS score calculation failed: {e}")
    
    async def generate_improvements_structured(
        self,
        resume: Resume,
        job_requirements: str,
        ats_score: ATSScore
    ) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions using structured output."""
        try:
            # Since we want a list, we'll create a wrapper model
            class ImprovementResult(BaseModel):
                suggestions: List[ImprovementSuggestion]
            
            structured_llm = self._create_structured_llm(ImprovementResult)
            
            prompt = self._create_prompt_template(
                system_message="You are a professional resume coach providing actionable improvement advice.",
                human_message="""
                Provide comprehensive improvement recommendations:
                
                Resume Analysis: {resume_analysis}
                Job Requirements: {job_requirements}
                ATS Analysis: {ats_analysis}
                
                Focus on actionable, specific recommendations that will have measurable impact.
                """
            )
            
            chain = prompt | structured_llm
            result = await chain.ainvoke({
                "resume_analysis": resume.model_dump_json(indent=2),
                "job_requirements": job_requirements,
                "ats_analysis": ats_score.model_dump_json(indent=2)
            })
            
            logger.info(f"Generated {len(result.suggestions)} improvement suggestions")
            return result.suggestions
            
        except Exception as e:
            logger.error(f"Improvement generation failed: {e}")
            raise OpenAIServiceError(f"Improvement generation failed: {e}")
    
    async def batch_process_structured(
        self, 
        requests: List[Dict[str, Any]], 
        max_concurrent: int = 5
    ) -> List[Any]:
        """Process multiple structured requests concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(request):
            async with semaphore:
                method_name = request.get("method")
                args = request.get("args", {})
                
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    return await method(**args)
                else:
                    raise OpenAIServiceError(f"Unknown method: {method_name}")
        
        logger.info(f"Processing {len(requests)} structured requests with max {max_concurrent} concurrent")
        
        tasks = [process_single(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def _get_extracted_sections(self, resume: Resume) -> List[str]:
        """Get list of sections that were successfully extracted."""
        sections = []
        
        if resume.personal_info.name:
            sections.append("personal_info")
        if resume.summary:
            sections.append("summary")
        if resume.experience:
            sections.append("experience")
        if resume.education:
            sections.append("education")
        if resume.skills.get_all_skills():
            sections.append("skills")
        if resume.certifications:
            sections.append("certifications")
        if resume.projects:
            sections.append("projects")
        
        return sections
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        return self.usage_tracker.get_summary()
    
    def reset_usage_tracking(self):
        """Reset usage tracking counters."""
        self.usage_tracker = UsageTracker()
        logger.info("Usage tracking reset")


# Global service instance
langgraph_openai_service = LangGraphOpenAIService()


# Convenience functions for common operations
async def parse_resume_structured_async(resume_text: str) -> ResumeAnalysisResult:
    """Convenience function for async structured resume parsing."""
    return await langgraph_openai_service.parse_resume_structured(resume_text)


def parse_resume_structured_sync(resume_text: str) -> ResumeAnalysisResult:
    """Convenience function for sync structured resume parsing."""
    return langgraph_openai_service.parse_resume_structured_sync(resume_text)


async def get_ats_score_structured_async(resume_text: str, job_description: str) -> ATSScore:
    """Convenience function for async structured ATS scoring."""
    return await langgraph_openai_service.calculate_ats_score_structured(resume_text, job_description)


async def get_job_match_structured_async(resume: Resume, job: Job) -> JobMatch:
    """Convenience function for async structured job matching."""
    return await langgraph_openai_service.analyze_job_match_structured(resume, job)


async def extract_skills_async(resume_text: str) -> Skills:
    """Convenience function for async skills extraction."""
    return await langgraph_openai_service.extract_skills_structured(resume_text)