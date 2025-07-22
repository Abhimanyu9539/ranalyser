"""
Pydantic models for job-related data structures.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, validator
from enum import Enum


class JobType(str, Enum):
    """Job type enumeration."""
    FULL_TIME = "full-time"
    PART_TIME = "part-time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"


class ExperienceLevel(str, Enum):
    """Experience level enumeration."""
    ENTRY_LEVEL = "entry-level"
    JUNIOR = "junior"
    MID_LEVEL = "mid-level"
    SENIOR = "senior"
    LEAD = "lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class RemoteType(str, Enum):
    """Remote work type enumeration."""
    ON_SITE = "on-site"
    REMOTE = "remote"
    HYBRID = "hybrid"


class JobRequirements(BaseModel):
    """Job requirements structure."""
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    education_requirements: List[str] = Field(default_factory=list)
    experience_years: Optional[int] = None
    experience_years_max: Optional[int] = None
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


class SalaryInfo(BaseModel):
    """Salary information structure."""
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    currency: str = "USD"
    period: str = "yearly"  # yearly, monthly, hourly
    is_disclosed: bool = False
    
    def get_salary_range_str(self) -> str:
        """Get formatted salary range string."""
        if not self.is_disclosed or (not self.min_salary and not self.max_salary):
            return "Not disclosed"
        
        if self.min_salary and self.max_salary:
            return f"${self.min_salary:,.0f} - ${self.max_salary:,.0f} {self.period}"
        elif self.min_salary:
            return f"${self.min_salary:,.0f}+ {self.period}"
        elif self.max_salary:
            return f"Up to ${self.max_salary:,.0f} {self.period}"
        
        return "Competitive"


class Company(BaseModel):
    """Company information structure."""
    name: str
    description: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None  # e.g., "50-100 employees"
    website: Optional[HttpUrl] = None
    logo_url: Optional[HttpUrl] = None
    location: Optional[str] = None
    rating: Optional[float] = Field(None, ge=0, le=5)


class Job(BaseModel):
    """Job posting data structure."""
    id: str
    title: str
    company: Company
    location: str
    remote_type: RemoteType = RemoteType.ON_SITE
    job_type: JobType = JobType.FULL_TIME
    experience_level: Optional[ExperienceLevel] = None
    
    # Job details
    description: str
    requirements: JobRequirements = Field(default_factory=JobRequirements)
    responsibilities: List[str] = Field(default_factory=list)
    benefits: List[str] = Field(default_factory=list)
    salary: SalaryInfo = Field(default_factory=SalaryInfo)
    
    # Metadata
    posted_date: Optional[datetime] = None
    application_deadline: Optional[datetime] = None
    source: str  # e.g., "LinkedIn", "Indeed"
    source_url: Optional[HttpUrl] = None
    apply_url: Optional[HttpUrl] = None
    
    # Internal tracking
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    
    def get_all_skills(self) -> List[str]:
        """Get all required and preferred skills."""
        all_skills = []
        all_skills.extend(self.requirements.required_skills)
        all_skills.extend(self.requirements.preferred_skills)
        return list(set(all_skills))  # Remove duplicates
    
    def get_job_summary(self) -> str:
        """Get a brief summary of the job."""
        summary_parts = [
            f"{self.title} at {self.company.name}",
            f"Location: {self.location}",
            f"Type: {self.job_type.value.title()}"
        ]
        
        if self.experience_level:
            summary_parts.append(f"Level: {self.experience_level.value.title()}")
        
        if self.salary.is_disclosed:
            summary_parts.append(f"Salary: {self.salary.get_salary_range_str()}")
        
        return " | ".join(summary_parts)


class JobMatch(BaseModel):
    """Job matching result structure."""
    job: Job
    match_percentage: float = Field(ge=0, le=100)
    match_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Detailed matching scores
    skill_match_score: float = Field(default=0, ge=0, le=100)
    experience_match_score: float = Field(default=0, ge=0, le=100)
    education_match_score: float = Field(default=0, ge=0, le=100)
    location_match_score: float = Field(default=0, ge=0, le=100)
    
    # Specific match analysis
    matched_skills: List[str] = Field(default_factory=list)
    missing_required_skills: List[str] = Field(default_factory=list)
    missing_preferred_skills: List[str] = Field(default_factory=list)
    
    # Recommendations
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Ranking info
    rank: Optional[int] = None
    confidence_score: float = Field(default=0, ge=0, le=1)
    
    def get_match_summary(self) -> str:
        """Get a summary of the match."""
        return (
            f"{self.match_percentage:.1f}% match for {self.job.title} "
            f"at {self.job.company.name}"
        )


class JobSearchCriteria(BaseModel):
    """Job search criteria structure."""
    keywords: List[str] = Field(default_factory=list)
    job_titles: List[str] = Field(default_factory=list)
    companies: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    remote_type: Optional[RemoteType] = None
    job_type: Optional[JobType] = None
    experience_level: Optional[ExperienceLevel] = None
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    required_skills: List[str] = Field(default_factory=list)
    exclude_companies: List[str] = Field(default_factory=list)
    date_posted_days: Optional[int] = 30  # Jobs posted within X days
    
    def to_search_query(self) -> str:
        """Convert criteria to search query string."""
        query_parts = []
        
        if self.keywords:
            query_parts.extend(self.keywords)
        
        if self.job_titles:
            query_parts.extend([f'"{title}"' for title in self.job_titles])
        
        if self.required_skills:
            query_parts.extend(self.required_skills)
        
        return " ".join(query_parts)


class JobSearchResult(BaseModel):
    """Job search result container."""
    jobs: List[Job] = Field(default_factory=list)
    total_found: int = 0
    search_criteria: JobSearchCriteria
    search_timestamp: datetime = Field(default_factory=datetime.now)
    source: str
    
    # Search metadata
    search_duration: Optional[float] = None
    pages_searched: int = 0
    api_calls_made: int = 0
    
    def get_summary(self) -> str:
        """Get search result summary."""
        return (
            f"Found {len(self.jobs)} jobs out of {self.total_found} total "
            f"from {self.source} in {self.search_duration or 0:.2f}s"
        )