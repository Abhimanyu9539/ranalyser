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