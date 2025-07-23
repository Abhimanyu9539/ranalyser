"""
OpenAI API integration service for the Resume Analyzer application.
"""
import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from openai import AsyncOpenAI, OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import get_openai_callback

from config.settings import settings
from config.prompts import ResumePrompts, JobMatchingPrompts, ATSPrompts, ImprovementPrompts
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIServiceError(Exception):
    """Custom exception for OpenAI service errors."""
    pass


class TokenUsageTracker:
    """Track token usage and costs across API calls."""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.start_time = datetime.now()
    
    def add_usage(self, prompt_tokens: int, completion_tokens: int, model: str = "gpt-4"):
        """Add token usage from an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.call_count += 1
        
        # Calculate cost (approximate rates)
        cost_per_1k = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002}
        }
        
        rates = cost_per_1k.get(model, cost_per_1k["gpt-4"])
        call_cost = (prompt_tokens / 1000 * rates["prompt"] + 
                    completion_tokens / 1000 * rates["completion"])
        self.total_cost += call_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost": round(self.total_cost, 4),
            "api_calls": self.call_count,
            "duration_seconds": round(duration, 2),
            "tokens_per_second": round(self.total_tokens / max(duration, 1), 2)
        }


class OpenAIService:
    """Service for interacting with OpenAI API and LangChain."""
    
    def __init__(self):
        """Initialize OpenAI service with configuration."""
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        
        # Initialize clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            max_retries=3,
            request_timeout=120
        )
        
        # Usage tracking
        self.usage_tracker = TokenUsageTracker()
        
        logger.info(f"OpenAI Service initialized with model: {self.model}")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling common issues."""
        try:
            # Clean response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            # Parse JSON
            return json.loads(cleaned.strip())
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response content: {response[:500]}...")
            
            # Try to extract JSON from response
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    extracted = response[start:end]
                    return json.loads(extracted)
            except:
                pass
            
            raise OpenAIServiceError(f"Failed to parse JSON response: {e}")
    
    async def _make_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make async completion call to OpenAI API."""
        try:
            response = await self.async_client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or 4000
            )
            
            # Track usage
            usage = response.usage
            self.usage_tracker.add_usage(
                usage.prompt_tokens,
                usage.completion_tokens,
                model or self.model
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIServiceError(f"API call failed: {e}")
    
    def _make_completion_sync(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make synchronous completion call to OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or 4000
            )
            
            # Track usage
            usage = response.usage
            self.usage_tracker.add_usage(
                usage.prompt_tokens,
                usage.completion_tokens,
                model or self.model
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIServiceError(f"API call failed: {e}")
    
    async def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Parse resume text into structured data."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert resume parser. Extract structured information accurately and return valid JSON."
            },
            {
                "role": "user",
                "content": ResumePrompts.PARSE_RESUME.format(resume_text=resume_text)
            }
        ]
        
        logger.info("Parsing resume with OpenAI...")
        start_time = time.time()
        
        response = await self._make_completion(messages, temperature=0.1)
        parsed_data = self._parse_json_response(response["content"])
        
        processing_time = time.time() - start_time
        logger.info(f"Resume parsed in {processing_time:.2f} seconds")
        
        return {
            "parsed_data": parsed_data,
            "processing_time": processing_time,
            "usage": response["usage"]
        }
    
    async def extract_skills(self, resume_text: str) -> Dict[str, Any]:
        """Extract and categorize skills from resume text."""
        messages = [
            {
                "role": "system",
                "content": "You are a skills extraction expert. Categorize skills accurately and return valid JSON."
            },
            {
                "role": "user",
                "content": ResumePrompts.EXTRACT_SKILLS.format(resume_text=resume_text)
            }
        ]
        
        logger.info("Extracting skills with OpenAI...")
        response = await self._make_completion(messages, temperature=0.1)
        skills_data = self._parse_json_response(response["content"])
        
        return {
            "skills": skills_data,
            "usage": response["usage"]
        }
    
    async def analyze_job_match(
        self, 
        resume_data: Dict[str, Any], 
        job_description: str
    ) -> Dict[str, Any]:
        """Analyze how well a resume matches a job description."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert career consultant analyzing job-resume compatibility."
            },
            {
                "role": "user",
                "content": JobMatchingPrompts.ANALYZE_JOB_MATCH.format(
                    resume_data=json.dumps(resume_data, indent=2),
                    job_description=job_description
                )
            }
        ]
        
        logger.info("Analyzing job match with OpenAI...")
        response = await self._make_completion(messages, temperature=0.2)
        match_analysis = self._parse_json_response(response["content"])
        
        return {
            "match_analysis": match_analysis,
            "usage": response["usage"]
        }
    
    async def calculate_ats_score(
        self, 
        resume_text: str, 
        job_description: str
    ) -> Dict[str, Any]:
        """Calculate ATS compatibility score for resume against job."""
        messages = [
            {
                "role": "system",
                "content": "You are an ATS expert analyzing resume compatibility with tracking systems."
            },
            {
                "role": "user",
                "content": ATSPrompts.CALCULATE_ATS_SCORE.format(
                    resume_text=resume_text,
                    job_description=job_description
                )
            }
        ]
        
        logger.info("Calculating ATS score with OpenAI...")
        response = await self._make_completion(messages, temperature=0.1)
        ats_analysis = self._parse_json_response(response["content"])
        
        return {
            "ats_score": ats_analysis,
            "usage": response["usage"]
        }
    
    async def generate_improvements(
        self,
        resume_analysis: Dict[str, Any],
        job_requirements: str,
        ats_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate improvement recommendations for resume."""
        messages = [
            {
                "role": "system",
                "content": "You are a professional resume coach providing actionable improvement advice."
            },
            {
                "role": "user",
                "content": ImprovementPrompts.GENERATE_IMPROVEMENTS.format(
                    resume_analysis=json.dumps(resume_analysis, indent=2),
                    job_requirements=job_requirements,
                    ats_analysis=json.dumps(ats_analysis, indent=2)
                )
            }
        ]
        
        logger.info("Generating improvements with OpenAI...")
        response = await self._make_completion(messages, temperature=0.3)
        improvements = self._parse_json_response(response["content"])
        
        return {
            "improvements": improvements,
            "usage": response["usage"]
        }
    
    def parse_resume_sync(self, resume_text: str) -> Dict[str, Any]:
        """Synchronous version of resume parsing."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert resume parser. Extract structured information accurately and return valid JSON."
            },
            {
                "role": "user",
                "content": ResumePrompts.PARSE_RESUME.format(resume_text=resume_text)
            }
        ]
        
        logger.info("Parsing resume with OpenAI (sync)...")
        start_time = time.time()
        
        response = self._make_completion_sync(messages, temperature=0.1)
        parsed_data = self._parse_json_response(response["content"])
        
        processing_time = time.time() - start_time
        logger.info(f"Resume parsed in {processing_time:.2f} seconds")
        
        return {
            "parsed_data": parsed_data,
            "processing_time": processing_time,
            "usage": response["usage"]
        }
    
    def use_langchain_llm(self, messages: List[Union[HumanMessage, SystemMessage, AIMessage]]) -> str:
        """Use LangChain LLM for more complex workflows."""
        try:
            with get_openai_callback() as cb:
                response = self.llm.invoke(messages)
                
                # Track usage
                self.usage_tracker.add_usage(
                    cb.prompt_tokens,
                    cb.completion_tokens,
                    self.model
                )
                
                logger.info(f"LangChain call - Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
                
                return response.content
                
        except Exception as e:
            logger.error(f"LangChain LLM error: {e}")
            raise OpenAIServiceError(f"LangChain call failed: {e}")
    
    async def batch_process(
        self, 
        requests: List[Dict[str, Any]], 
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple requests concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(request):
            async with semaphore:
                method_name = request.get("method")
                args = request.get("args", {})
                
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    if asyncio.iscoroutinefunction(method):
                        return await method(**args)
                    else:
                        return method(**args)
                else:
                    raise OpenAIServiceError(f"Unknown method: {method_name}")
        
        logger.info(f"Processing {len(requests)} requests with max {max_concurrent} concurrent")
        
        tasks = [process_single(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        return self.usage_tracker.get_summary()
    
    def reset_usage_tracking(self):
        """Reset usage tracking counters."""
        self.usage_tracker = TokenUsageTracker()
        logger.info("Usage tracking reset")


# Global service instance
openai_service = OpenAIService()


# Convenience functions for common operations
async def parse_resume_async(resume_text: str) -> Dict[str, Any]:
    """Convenience function for async resume parsing."""
    return await openai_service.parse_resume(resume_text)


def parse_resume_sync(resume_text: str) -> Dict[str, Any]:
    """Convenience function for sync resume parsing."""
    return openai_service.parse_resume_sync(resume_text)


async def get_ats_score_async(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Convenience function for async ATS scoring."""
    return await openai_service.calculate_ats_score(resume_text, job_description)


async def get_job_match_async(resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
    """Convenience function for async job matching."""
    return await openai_service.analyze_job_match(resume_data, job_description)