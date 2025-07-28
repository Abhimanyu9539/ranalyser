"""
Pydantic Models for resume data structure
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr, HttpUrl, field_validator
from enum import Enum

class SkillCategory(str, Enum):
    """Enumeration of skill categories"""
    TECHNICAL = "technical_skills"
    PROGRAMMING = "programming_languages"
    FRAMEWORKS = "frameworks_libraries"
    TOOLS = "tools_software"
    DATABASES = "databases"
    CLOUD = "cloud_platforms"
    SOFT_SKILLS = "soft_skills"
    DOMAIN = "domain_expertise"

class PersonalInfo(BaseModel):
    """Personal Information from Resume"""
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[HttpUrl] = None
    portfolio: Optional[HttpUrl] = None
    github: Optional[HttpUrl] = None

class WorkExperience(BaseModel):
    """Work Experience from Resume"""
    title: str
    company: str
    location: Optional[str] = None
    start_date: str
    end_date: str
    description: str
    key_achievements: List[str] = Field(default_factory=list)
    technologies_used: List[str] = Field(default_factory=list)

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        """Accept dates as-is - both YYYY and MM/YYYY formats are valid."""
        if not v:
            return v
            
        v = v.strip()
        
        # Handle "Present" and variations
        if v.lower() in ['present', 'current', 'now', 'ongoing']:
            return "Present"
        
        # Accept MM/YYYY format
        try:
            datetime.strptime(v, '%m/%Y')
            return v  # Valid MM/YYYY - keep as is
        except ValueError:
            pass
        
        # Accept YYYY format  
        try:
            datetime.strptime(v, '%Y')
            return v  # Valid YYYY - keep as is (don't convert!)
        except ValueError:
            pass
        
        # If neither format works, return as-is (don't raise error)
        return v


class Education(BaseModel):
    """Education Entry"""
    degree: str
    field: str
    institution: str
    graduation_date: Optional[str] = None  # MM/YYYY format
    gpa: Optional[str] = None
    honors: Optional[str] = None

    @field_validator('graduation_date')
    @classmethod
    def validate_graduation_date(cls, v):
        """Validate graduation date format."""
        if v is None:
            return v
        try:
            datetime.strptime(v, '%m/%Y')
            return v
        except ValueError:
            try:
                datetime.strptime(v, '%Y')
                return f"01/{v}"  # Convert YYYY to MM/YYYY
            except ValueError:
                raise ValueError('Graduation date must be in MM/YYYY format')


class Certification(BaseModel):
    """Certification entry."""
    name: str
    issuer: str
    date: Optional[str] = None  # MM/YYYY format
    expiry: Optional[str] = None  # MM/YYYY format or "Never"
    credential_id: Optional[str] = None
    url: Optional[HttpUrl] = None


class Project(BaseModel): 
    """Project entry."""
    name: str
    description: str
    technologies: List[str] = Field(default_factory=list)
    url: Optional[HttpUrl] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    key_features: List[str] = Field(default_factory=list)


class Skills(BaseModel):
    """Skills Categorized by Type"""
    technical_skills: List[str] = Field(default_factory=list)
    programming_languages: List[str] = Field(default_factory=list)
    frameworks_libraries: List[str] = Field(default_factory=list)
    tools_software: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    cloud_platforms: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    domain_expertise: List[str] = Field(default_factory=list)

    def get_all_skills(self) -> List[str]:
        """Get all skills from all categories as a flat list."""
        all_skills = []
        for skills_list in [
            self.technical_skills, self.programming_languages,
            self.frameworks_libraries, self.tools_software,
            self.databases, self.cloud_platforms, self.soft_skills,
            self.domain_expertise
        ]:
            all_skills.extend(skills_list)
        return list(set(all_skills))  # Removes duplicates

    def get_technical_skills(self) -> List[str]:
        """Get only technical skills (excluding soft skills)"""
        technical = []
        for skill_list in [
            self.technical_skills, self.programming_languages,
            self.frameworks_libraries, self.tools_software,
            self.databases, self.cloud_platforms, self.domain_expertise
        ]:
            technical.extend(skill_list)
        return list(set(technical))

class Resume(BaseModel):
    """Complete resume data structure."""
    id: Optional[str] = None
    personal_info: PersonalInfo
    summary: Optional[str] = None
    experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: List[Certification] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    file_path: Optional[str] = None
    original_filename: Optional[str] = None
    
    def get_years_of_experience(self) -> float:
        """Calculate total years of experience."""
        total_months = 0
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        for exp in self.experience:
            try:
                start_parts = exp.start_date.split('/')
                start_month, start_year = int(start_parts[0]), int(start_parts[1])
                
                if exp.end_date.lower() == 'present':
                    end_month, end_year = current_month, current_year
                else:
                    end_parts = exp.end_date.split('/')
                    end_month, end_year = int(end_parts[0]), int(end_parts[1])
                
                months = (end_year - start_year) * 12 + (end_month - start_month)
                total_months += max(0, months)
                
            except (ValueError, IndexError):
                continue  # Skip invalid dates
        
        return round(total_months / 12, 1)
    
    def get_latest_position(self) -> Optional[WorkExperience]:
        """Get the most recent work experience."""
        if not self.experience:
            return None
        
        # Sort by end date, with "Present" positions first
        sorted_exp = sorted(
            self.experience,
            key=lambda x: (x.end_date.lower() != 'present', x.end_date),
            reverse=True
        )
        return sorted_exp[0]
    
    def get_education_level(self) -> str:
        """Get the highest education level."""
        if not self.education:
            return "Not specified"
        
        degree_hierarchy = {
            'phd': 4, 'doctorate': 4, 'doctoral': 4,
            'master': 3, 'mba': 3, 'ms': 3, 'ma': 3, 'mg': 3,
            'bachelor': 2, 'bs': 2, 'ba': 2, 'btech': 2,
            'associate': 1, 'diploma': 1, 'certificate': 1
        }
        
        highest_level = 0
        highest_degree = "Not specified"
        
        for edu in self.education:
            degree_lower = edu.degree.lower()
            for key, level in degree_hierarchy.items():
                if key in degree_lower and level > highest_level:
                    highest_level = level
                    highest_degree = edu.degree
        
        return highest_degree
    
    def to_search_string(self) -> str:
        """Convert resume to searchable string for job matching."""
        search_parts = []
        
        # Add summary
        if self.summary:
            search_parts.append(self.summary)
        
        # Add experience descriptions
        for exp in self.experience:
            search_parts.extend([exp.title, exp.company, exp.description])
            search_parts.extend(exp.key_achievements)
        
        # Add skills
        search_parts.extend(self.skills.get_all_skills())
        
        # Add education
        for edu in self.education:
            search_parts.extend([edu.degree, edu.field or "", edu.institution])
        
        return " ".join(filter(None, search_parts))


class ResumeAnalysisResult(BaseModel):
    """Result of Resume Analysis process"""
    resume: Resume
    parsing_confidence: float = Field(ge=0, le=1)
    extracted_sections: List[str] = Field(default_factory=list)
    parsing_issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    # Processing metadata
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)