"""
Main ATS Scorer interface - orchestrates ATS analysis components.
This is the primary entry point for ATS scoring functionality.
"""
import logging
import time
from typing import Dict, Any, Optional

from src.models.resume import Resume
from src.models.job import Job
from src.models.ats_score import ATSScore
from config.settings import settings

# Import components (will be created next)
# from .ats_validator import ATSValidator
# from .ats_formatter import ATSDataFormatter  
# from .ats_analyzer import ATSAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATSScoringError(Exception):
    """Custom exception for ATS scoring errors."""
    pass


class ATSScoringResult:
    """ATS scoring result with metadata and additional context."""
    
    def __init__(self, ats_score: ATSScore, metadata: Dict[str, Any]):
        self.ats_score = ats_score
        self.metadata = metadata
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the ATS scoring results."""
        return {
            "overall_score": self.ats_score.overall_score,
            "rating": self.ats_score.rating.value,
            "top_strengths": self.ats_score.top_strengths[:3],
            "critical_issues": self.ats_score.critical_issues[:3],
            "processing_time": self.metadata.get("processing_time", 0)
        }
    
    def get_improvement_priorities(self) -> Dict[str, Any]:
        """Get prioritized improvement suggestions."""
        suggestions = self.ats_score.get_improvement_priority_list()
        
        return {
            "critical": [s for s in suggestions if s.priority == "critical"],
            "high": [s for s in suggestions if s.priority == "high"],
            "medium": [s for s in suggestions if s.priority == "medium"],
            "low": [s for s in suggestions if s.priority == "low"]
        }


class LLMATSScorer:
    """
    Main ATS scorer that orchestrates validation, formatting, and analysis
    to provide comprehensive resume compatibility scoring.
    """
    
    def __init__(self):
        """Initialize the ATS scorer with all required components."""
        # Initialize components (will uncomment as we create them)
        # self.validator = ATSValidator()
        # self.formatter = ATSDataFormatter()
        # self.analyzer = ATSAnalyzer()
        
        logger.info("LLM ATS Scorer initialized")
    
    async def calculate_ats_score(
        self, 
        resume: Resume, 
        job: Job,
        include_metadata: bool = True
    ) -> ATSScoringResult:
        """
        Calculate comprehensive ATS score using structured resume data.
        
        This is the main public method that orchestrates the entire ATS analysis process.
        
        Args:
            resume: Structured resume data (already extracted by other tools)
            job: Job posting data with requirements
            include_metadata: Whether to include detailed processing metadata
        
        Returns:
            ATSScoringResult with comprehensive ATS analysis and metadata
        
        Raises:
            ATSScoringError: If scoring fails due to invalid inputs or processing errors
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting ATS analysis for {job.title} at {job.company.name}")
            
            # Step 1: Validate inputs
            self._validate_inputs(resume, job)
            
            # Step 2: Format data for analysis
            formatted_data = self._format_data_for_analysis(resume, job)
            
            # Step 3: Perform LLM-based analysis
            ats_score = await self._perform_analysis(formatted_data)
            
            # Step 4: Generate metadata
            processing_time = time.time() - start_time
            metadata = self._generate_metadata(resume, job, ats_score, processing_time) if include_metadata else {}
            
            result = ATSScoringResult(ats_score, metadata)
            
            logger.info(f"ATS analysis completed: {ats_score.overall_score}/100 ({ats_score.rating.value}) in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"ATS scoring failed: {e}")
            raise ATSScoringError(f"Failed to calculate ATS score: {e}")
    
    async def quick_ats_assessment(self, resume: Resume, job: Job) -> Dict[str, Any]:
        """
        Perform a quick ATS compatibility assessment with essential metrics only.
        
        This method provides faster analysis with core compatibility indicators,
        ideal for bulk processing or initial screening.
        
        Args:
            resume: Structured resume data
            job: Job posting data
        
        Returns:
            Dictionary with essential ATS metrics and quick recommendations
        """
        try:
            logger.info("Performing quick ATS assessment")
            
            # Use main analysis but return simplified results
            result = await self.calculate_ats_score(resume, job, include_metadata=False)
            
            return {
                "overall_score": result.ats_score.overall_score,
                "rating": result.ats_score.rating.value,
                "keyword_match_percentage": result.ats_score.keyword_analysis.get_keyword_match_percentage(),
                "top_strengths": result.ats_score.top_strengths[:2],
                "critical_issues": result.ats_score.critical_issues[:2],
                "quick_wins": result.ats_score.quick_wins[:2],
                "missing_required_skills": result.ats_score.keyword_analysis.missing_required_keywords[:5],
                "recommendation": self._get_quick_recommendation(result.ats_score.overall_score)
            }
            
        except Exception as e:
            logger.error(f"Quick ATS assessment failed: {e}")
            raise ATSScoringError(f"Quick assessment failed: {e}")
    
    def validate_resume_for_ats(self, resume: Resume) -> Dict[str, Any]:
        """
        Validate resume structure and completeness for ATS compatibility.
        
        Args:
            resume: Resume object to validate
        
        Returns:
            Validation report with issues and recommendations
        """
        try:
            # This will use ATSValidator when implemented
            validation_report = {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "recommendations": []
            }
            
            # Basic validation logic (placeholder)
            if not resume.personal_info.name:
                validation_report["issues"].append("Missing name in personal information")
                validation_report["is_valid"] = False
            
            if not resume.personal_info.email:
                validation_report["issues"].append("Missing email address")
                validation_report["is_valid"] = False
            
            if not resume.experience:
                validation_report["warnings"].append("No work experience found")
            
            if not resume.skills.get_all_skills():
                validation_report["warnings"].append("No skills listed")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Resume validation failed: {e}")
            return {
                "is_valid": False,
                "issues": [f"Validation error: {e}"],
                "warnings": [],
                "recommendations": []
            }
    
    def _validate_inputs(self, resume: Resume, job: Job) -> None:
        """Validate that resume and job inputs are suitable for analysis."""
        # Placeholder - will use ATSValidator
        if not resume.personal_info.name:
            raise ATSScoringError("Resume must have a name")
        
        if not job.title:
            raise ATSScoringError("Job must have a title")
        
        if not job.description:
            raise ATSScoringError("Job must have a description")
    
    def _format_data_for_analysis(self, resume: Resume, job: Job) -> Dict[str, Any]:
        """Format resume and job data for LLM analysis."""
        # Placeholder - will use ATSDataFormatter
        return {
            "resume": resume,
            "job": job,
            "formatted_resume_text": f"Resume for {resume.personal_info.name}",
            "formatted_job_text": f"Job: {job.title} at {job.company.name}"
        }
    
    async def _perform_analysis(self, formatted_data: Dict[str, Any]) -> ATSScore:
        """Perform the actual LLM-based ATS analysis."""
        # Placeholder - will use ATSAnalyzer
        from src.models.ats_score import ATSScore, ATSRating
        
        # Create a basic ATSScore (placeholder)
        return ATSScore(
            overall_score=75.0,
            rating=ATSRating.GOOD,
            top_strengths=["Strong technical skills", "Relevant experience"],
            critical_issues=["Missing quantified achievements"],
            quick_wins=["Add professional summary", "Include more keywords"]
        )
    
    def _generate_metadata(self, resume: Resume, job: Job, ats_score: ATSScore, processing_time: float) -> Dict[str, Any]:
        """Generate comprehensive metadata for the ATS analysis."""
        return {
            "processing_time": processing_time,
            "job_title": job.title,
            "job_company": job.company.name,
            "resume_name": resume.personal_info.name,
            "overall_score": ats_score.overall_score,
            "rating": ats_score.rating.value,
            "analysis_timestamp": time.time(),
            "model_used": settings.openai_model,
            "analysis_method": "llm_structured_output",
            "keyword_matches": len(ats_score.keyword_analysis.required_keywords_found) if ats_score.keyword_analysis else 0,
            "improvement_suggestions_count": len(ats_score.improvement_suggestions),
            "critical_issues_count": len(ats_score.critical_issues),
            "resume_sections_analyzed": {
                "personal_info": bool(resume.personal_info.name),
                "summary": bool(resume.summary),
                "experience": bool(resume.experience),
                "education": bool(resume.education),
                "skills": bool(resume.skills.get_all_skills()),
                "projects": bool(resume.projects),
                "certifications": bool(resume.certifications)
            }
        }
    
    def _get_quick_recommendation(self, score: float) -> str:
        """Get a quick recommendation based on overall score."""
        if score >= 90:
            return "Excellent ATS compatibility - resume is well-optimized"
        elif score >= 75:
            return "Good ATS compatibility - minor improvements recommended"
        elif score >= 60:
            return "Fair ATS compatibility - several improvements needed"
        else:
            return "Poor ATS compatibility - major optimization required"


# Global scorer instance for easy importing
llm_ats_scorer = LLMATSScorer()


# Convenience functions for common use cases
async def calculate_ats_score(resume: Resume, job: Job) -> ATSScoringResult:
    """Convenience function for calculating ATS score."""
    return await llm_ats_scorer.calculate_ats_score(resume, job)


async def quick_ats_check(resume: Resume, job: Job) -> Dict[str, Any]:
    """Convenience function for quick ATS assessment."""
    return await llm_ats_scorer.quick_ats_assessment(resume, job)


def validate_resume(resume: Resume) -> Dict[str, Any]:
    """Convenience function for resume validation."""
    return llm_ats_scorer.validate_resume_for_ats(resume)