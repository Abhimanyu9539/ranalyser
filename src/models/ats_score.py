"""
Pydantic models for ATS (Applicant Tracking System) scoring.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class ATSRating(str, Enum):
    """ATS compatibility rating levels."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    FAIR = "fair"           # 60-74
    POOR = "poor"           # 0-59


class ScoreCategory(str, Enum):
    """ATS scoring categories."""
    KEYWORD_OPTIMIZATION = "keyword_optimization"
    FORMAT_STRUCTURE = "format_structure"
    CONTENT_QUALITY = "content_quality"
    TECHNICAL_COMPATIBILITY = "technical_compatibility"
    LENGTH_CONCISENESS = "length_conciseness"


class KeywordAnalysis(BaseModel):
    """Keyword matching and optimization analysis."""
    total_keywords_found: int = 0
    required_keywords_found: List[str] = Field(default_factory=list)
    missing_required_keywords: List[str] = Field(default_factory=list)
    preferred_keywords_found: List[str] = Field(default_factory=list)
    missing_preferred_keywords: List[str] = Field(default_factory=list)
    keyword_density: float = Field(default=0.0, ge=0, le=1)  # Percentage of text that is keywords
    keyword_frequency: Dict[str, int] = Field(default_factory=dict)  # How often each keyword appears
    context_relevance: float = Field(default=0.0, ge=0, le=1)  # How well keywords fit context
    
    def get_keyword_match_percentage(self) -> float:
        """Calculate overall keyword match percentage."""
        total_required = len(self.required_keywords_found) + len(self.missing_required_keywords)
        if total_required == 0:
            return 100.0
        return (len(self.required_keywords_found) / total_required) * 100


class FormatAnalysis(BaseModel):
    """Resume format and structure analysis."""
    has_contact_info: bool = True
    has_professional_summary: bool = False
    has_work_experience: bool = True
    has_education: bool = True
    has_skills_section: bool = True
    
    # Format quality indicators
    consistent_formatting: bool = True
    proper_bullet_points: bool = True
    clear_section_headers: bool = True
    appropriate_font_usage: bool = True
    proper_spacing: bool = True
    
    # Structure scores
    section_organization_score: float = Field(default=0, ge=0, le=100)
    readability_score: float = Field(default=0, ge=0, le=100)
    visual_appeal_score: float = Field(default=0, ge=0, le=100)
    
    # Issues found
    formatting_issues: List[str] = Field(default_factory=list)
    structural_issues: List[str] = Field(default_factory=list)


class ContentAnalysis(BaseModel):
    """Content quality and relevance analysis."""
    quantified_achievements_count: int = 0
    action_verbs_count: int = 0
    total_achievements: int = 0
    
    # Content quality metrics
    achievement_quantification_rate: float = Field(default=0, ge=0, le=1)
    action_verb_usage_score: float = Field(default=0, ge=0, le=100)
    relevance_to_job_score: float = Field(default=0, ge=0, le=100)
    experience_alignment_score: float = Field(default=0, ge=0, le=100)
    
    # Content strengths and weaknesses
    strong_points: List[str] = Field(default_factory=list)
    weak_points: List[str] = Field(default_factory=list)
    missing_content_areas: List[str] = Field(default_factory=list)


class TechnicalCompatibility(BaseModel):
    """Technical ATS compatibility analysis."""
    is_machine_readable: bool = True
    has_complex_formatting: bool = False
    has_images_or_graphics: bool = False
    has_tables: bool = False
    uses_standard_fonts: bool = True
    
    # Parsing quality
    text_extraction_quality: float = Field(default=100, ge=0, le=100)
    character_encoding_issues: int = 0
    parsing_errors: List[str] = Field(default_factory=list)
    
    # File format info
    file_format: Optional[str] = None
    file_size_kb: Optional[float] = None
    page_count: Optional[int] = None


class LengthAnalysis(BaseModel):
    """Resume length and conciseness analysis."""
    total_pages: int = 1
    total_words: int = 0
    total_characters: int = 0
    
    # Length appropriateness
    is_appropriate_length: bool = True
    length_recommendation: str = "Good length"
    
    # Conciseness metrics
    average_sentence_length: float = 0
    redundancy_score: float = Field(default=0, ge=0, le=100)  # Lower is better
    conciseness_score: float = Field(default=100, ge=0, le=100)  # Higher is better
    
    # Specific length issues
    sections_too_long: List[str] = Field(default_factory=list)
    sections_too_short: List[str] = Field(default_factory=list)


class CategoryScore(BaseModel):
    """Individual category scoring details."""
    category: ScoreCategory
    score: float = Field(ge=0, le=100)
    max_possible_score: float = 100
    weight: float = Field(gt=0, le=1)  # Weight in overall score calculation
    
    # Detailed feedback
    feedback: str = ""
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    
    # Sub-scores for this category
    sub_scores: Dict[str, float] = Field(default_factory=dict)
    
    def get_weighted_score(self) -> float:
        """Calculate weighted score contribution."""
        return (self.score / self.max_possible_score) * self.weight * 100


class ImprovementSuggestion(BaseModel):
    """Individual improvement suggestion."""
    category: ScoreCategory
    priority: str = Field(regex="^(critical|high|medium|low)$")
    issue: str
    recommendation: str
    impact: str  # Expected impact description
    implementation: str  # How to implement the suggestion
    estimated_score_improvement: float = Field(default=0, ge=0, le=100)
    
    # Implementation details
    difficulty: str = Field(default="medium", regex="^(easy|medium|hard)$")
    time_estimate: str = "1-2 hours"
    resources_needed: List[str] = Field(default_factory=list)


class ATSScore(BaseModel):
    """Complete ATS scoring result."""
    # Overall scoring
    overall_score: float = Field(ge=0, le=100)
    rating: ATSRating
    
    # Category-wise scores
    category_scores: List[CategoryScore] = Field(default_factory=list)
    
    # Detailed analysis
    keyword_analysis: KeywordAnalysis = Field(default_factory=KeywordAnalysis)
    format_analysis: FormatAnalysis = Field(default_factory=FormatAnalysis)
    content_analysis: ContentAnalysis = Field(default_factory=ContentAnalysis)
    technical_compatibility: TechnicalCompatibility = Field(default_factory=TechnicalCompatibility)
    length_analysis: LengthAnalysis = Field(default_factory=LengthAnalysis)
    
    # Improvement suggestions
    improvement_suggestions: List[ImprovementSuggestion] = Field(default_factory=list)
    
    # Summary insights
    top_strengths: List[str] = Field(default_factory=list)
    critical_issues: List[str] = Field(default_factory=list)
    quick_wins: List[str] = Field(default_factory=list)  # Easy improvements with high impact
    
    # Metadata
    job_title: Optional[str] = None
    job_company: Optional[str] = None
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    model_version: Optional[str] = None
    processing_time: Optional[float] = None
    
    @validator('overall_score')
    def calculate_overall_score(cls, v, values):
        """Calculate overall score from category scores if not provided."""
        if v == 0 and 'category_scores' in values:
            category_scores = values['category_scores']
            if category_scores:
                total_weighted = sum(score.get_weighted_score() for score in category_scores)
                return round(total_weighted, 1)
        return v
    
    @validator('rating')
    def set_rating_from_score(cls, v, values):
        """Set rating based on overall score."""
        if 'overall_score' in values:
            score = values['overall_score']
            if score >= 90:
                return ATSRating.EXCELLENT
            elif score >= 75:
                return ATSRating.GOOD
            elif score >= 60:
                return ATSRating.FAIR
            else:
                return ATSRating.POOR
        return v
    
    def get_score_summary(self) -> str:
        """Get a brief score summary."""
        return f"ATS Score: {self.overall_score}/100 ({self.rating.value.title()})"
    
    def get_improvement_priority_list(self) -> List[ImprovementSuggestion]:
        """Get improvements sorted by priority and impact."""
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        return sorted(
            self.improvement_suggestions,
            key=lambda x: (
                priority_order.get(x.priority, 4),
                -x.estimated_score_improvement
            )
        )
    
    def get_category_score(self, category: ScoreCategory) -> Optional[CategoryScore]:
        """Get score for a specific category."""
        for score in self.category_scores:
            if score.category == category:
                return score
        return None
    
    def get_potential_max_score(self) -> float:
        """Calculate potential maximum score if all improvements are implemented."""
        current_score = self.overall_score
        potential_improvement = sum(
            suggestion.estimated_score_improvement 
            for suggestion in self.improvement_suggestions
        )
        return min(100.0, current_score + potential_improvement)


class ATSBenchmark(BaseModel):
    """ATS scoring benchmarks and standards."""
    industry: Optional[str] = None
    job_level: Optional[str] = None
    
    # Benchmark scores
    excellent_threshold: float = 90
    good_threshold: float = 75
    fair_threshold: float = 60
    
    # Category weights for this benchmark
    category_weights: Dict[ScoreCategory, float] = Field(
        default_factory=lambda: {
            ScoreCategory.KEYWORD_OPTIMIZATION: 0.30,
            ScoreCategory.FORMAT_STRUCTURE: 0.25,
            ScoreCategory.CONTENT_QUALITY: 0.20,
            ScoreCategory.TECHNICAL_COMPATIBILITY: 0.15,
            ScoreCategory.LENGTH_CONCISENESS: 0.10
        }
    )
    
    # Industry-specific considerations
    critical_keywords: List[str] = Field(default_factory=list)
    preferred_sections: List[str] = Field(default_factory=list)
    common_issues: List[str] = Field(default_factory=list)


class ATSComparison(BaseModel):
    """Comparison of ATS scores across multiple jobs or candidates."""
    scores: List[ATSScore] = Field(default_factory=list)
    benchmark: Optional[ATSBenchmark] = None
    
    # Comparison metrics
    average_score: float = 0
    highest_score: float = 0
    lowest_score: float = 0
    score_distribution: Dict[ATSRating, int] = Field(default_factory=dict)
    
    # Analysis
    common_strengths: List[str] = Field(default_factory=list)
    common_weaknesses: List[str] = Field(default_factory=list)
    improvement_trends: List[str] = Field(default_factory=list)
    
    def calculate_stats(self):
        """Calculate comparison statistics."""
        if not self.scores:
            return
        
        scores_values = [score.overall_score for score in self.scores]
        self.average_score = sum(scores_values) / len(scores_values)
        self.highest_score = max(scores_values)
        self.lowest_score = min(scores_values)
        
        # Calculate rating distribution
        rating_counts = {}
        for score in self.scores:
            rating_counts[score.rating] = rating_counts.get(score.rating, 0) + 1
        self.score_distribution = rating_counts