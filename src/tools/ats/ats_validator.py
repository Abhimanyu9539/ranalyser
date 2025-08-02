"""
ATS Validator - Input validation and data quality checks for ATS scoring.
Ensures Resume and Job objects have sufficient data for meaningful ATS analysis.
"""
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.models.resume import Resume, WorkExperience, Education
from src.models.job import Job
from pydantic import ValidationError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationIssue:
    """Represents a validation issue with severity and context."""
    
    def __init__(self, field: str, issue: str, severity: str, recommendation: str = ""):
        self.field = field
        self.issue = issue
        self.severity = severity  # 'critical', 'warning', 'info'
        self.recommendation = recommendation
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "field": self.field,
            "issue": self.issue,
            "severity": self.severity,
            "recommendation": self.recommendation
        }


class ValidationReport:
    """Comprehensive validation report for Resume or Job objects."""
    
    def __init__(self):
        self.is_valid = True
        self.issues: List[ValidationIssue] = []
        self.score = 100.0  # Validation quality score
        self.completeness_percentage = 0.0
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue and update overall validity."""
        self.issues.append(issue)
        
        if issue.severity == 'critical':
            self.is_valid = False
            self.score -= 20
        elif issue.severity == 'warning':
            self.score -= 5
        elif issue.severity == 'info':
            self.score -= 1
        
        self.score = max(0, self.score)
    
    def get_issues_by_severity(self, severity: str) -> List[ValidationIssue]:
        """Get all issues of a specific severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get all critical issues that prevent analysis."""
        return self.get_issues_by_severity('critical')
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning issues that may impact analysis quality."""
        return self.get_issues_by_severity('warning')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "validation_score": self.score,
            "completeness_percentage": self.completeness_percentage,
            "total_issues": len(self.issues),
            "critical_issues": [issue.to_dict() for issue in self.get_critical_issues()],
            "warnings": [issue.to_dict() for issue in self.get_warnings()],
            "info_items": [issue.to_dict() for issue in self.get_issues_by_severity('info')],
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> str:
        """Generate a human-readable summary of validation results."""
        if self.is_valid and len(self.issues) == 0:
            return "Resume/Job data is complete and ready for ATS analysis"
        elif self.is_valid:
            return f"Data is valid with {len(self.get_warnings())} minor issues"
        else:
            critical_count = len(self.get_critical_issues())
            return f"Data has {critical_count} critical issue(s) that must be resolved"


class ATSValidator:
    """
    Validates Resume and Job objects for ATS analysis compatibility.
    Checks data completeness, quality, and format requirements.
    """
    
    def __init__(self):
        """Initialize the validator with quality thresholds."""
        # Minimum requirements for ATS analysis
        self.min_work_experience_years = 0  # No minimum, but will warn if empty
        self.min_skills_count = 3
        self.min_description_length = 20
        
        # Email validation pattern
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        # Phone validation pattern (flexible)
        self.phone_pattern = re.compile(r'[\+]?[\d\s\-\(\)]{10,}')
        
        logger.info("ATS Validator initialized")
    
    def validate_resume(self, resume: Resume) -> ValidationReport:
        """
        Comprehensive validation of Resume object for ATS compatibility.
        
        Args:
            resume: Resume object to validate
        
        Returns:
            ValidationReport with detailed validation results
        """
        try:
            logger.debug(f"Validating resume for {resume.personal_info.name}")
            
            report = ValidationReport()
            
            # Validate each section
            self._validate_personal_info(resume.personal_info, report)
            self._validate_summary(resume.summary, report)
            self._validate_work_experience(resume.experience, report)
            self._validate_education(resume.education, report)
            self._validate_skills(resume.skills, report)
            self._validate_projects(resume.projects, report)
            self._validate_certifications(resume.certifications, report)
            
            # Calculate completeness percentage
            report.completeness_percentage = self._calculate_resume_completeness(resume)
            
            logger.debug(f"Resume validation completed. Valid: {report.is_valid}, Score: {report.score}")
            return report
            
        except Exception as e:
            logger.error(f"Resume validation failed: {e}")
            report = ValidationReport()
            report.add_issue(ValidationIssue(
                field="general",
                issue=f"Validation process failed: {str(e)}",
                severity="critical",
                recommendation="Check resume data structure and try again"
            ))
            return report
    
    def validate_job(self, job: Job) -> ValidationReport:
        """
        Validate Job object for ATS analysis requirements.
        
        Args:
            job: Job object to validate
        
        Returns:
            ValidationReport with job validation results
        """
        try:
            logger.debug(f"Validating job posting: {job.title} at {job.company.name}")
            
            report = ValidationReport()
            
            # Validate job basic information
            self._validate_job_basics(job, report)
            self._validate_job_requirements(job.requirements, report)
            self._validate_job_description(job.description, report)
            self._validate_company_info(job.company, report)
            
            # Calculate job completeness
            report.completeness_percentage = self._calculate_job_completeness(job)
            
            logger.debug(f"Job validation completed. Valid: {report.is_valid}, Score: {report.score}")
            return report
            
        except Exception as e:
            logger.error(f"Job validation failed: {e}")
            report = ValidationReport()
            report.add_issue(ValidationIssue(
                field="general",
                issue=f"Job validation failed: {str(e)}",
                severity="critical",
                recommendation="Check job data structure and format"
            ))
            return report
    
    def validate_for_ats_analysis(self, resume: Resume, job: Job) -> Tuple[ValidationReport, ValidationReport]:
        """
        Validate both resume and job for ATS analysis compatibility.
        
        Args:
            resume: Resume object to validate
            job: Job object to validate
        
        Returns:
            Tuple of (resume_report, job_report)
        """
        resume_report = self.validate_resume(resume)
        job_report = self.validate_job(job)
        
        logger.info(f"Combined validation: Resume valid: {resume_report.is_valid}, Job valid: {job_report.is_valid}")
        
        return resume_report, job_report
    
    def can_perform_ats_analysis(self, resume: Resume, job: Job) -> Tuple[bool, List[str]]:
        """
        Quick check if ATS analysis can be performed with given data.
        
        Args:
            resume: Resume object
            job: Job object
        
        Returns:
            Tuple of (can_analyze, blocking_issues)
        """
        blocking_issues = []
        
        # Critical resume requirements
        if not resume.personal_info.name:
            blocking_issues.append("Resume must have a name")
        
        if not resume.experience and not resume.projects:
            blocking_issues.append("Resume must have either work experience or projects")
        
        if not resume.skills.get_all_skills():
            blocking_issues.append("Resume must have skills listed")
        
        # Critical job requirements
        if not job.title:
            blocking_issues.append("Job must have a title")
        
        if not job.description or len(job.description.strip()) < 50:
            blocking_issues.append("Job must have a meaningful description")
        
        can_analyze = len(blocking_issues) == 0
        
        return can_analyze, blocking_issues
    
    # Private validation methods for each section
    def _validate_personal_info(self, personal_info, report: ValidationReport):
        """Validate personal information section."""
        if not personal_info.name or len(personal_info.name.strip()) < 2:
            report.add_issue(ValidationIssue(
                field="personal_info.name",
                issue="Name is missing or too short",
                severity="critical",
                recommendation="Add full name to resume"
            ))
        
        if not personal_info.email:
            report.add_issue(ValidationIssue(
                field="personal_info.email",
                issue="Email address is missing",
                severity="critical", 
                recommendation="Add professional email address"
            ))
        elif not self.email_pattern.match(str(personal_info.email)):
            report.add_issue(ValidationIssue(
                field="personal_info.email",
                issue="Email format appears invalid",
                severity="warning",
                recommendation="Verify email address format"
            ))
        
        if not personal_info.phone:
            report.add_issue(ValidationIssue(
                field="personal_info.phone",
                issue="Phone number is missing",
                severity="warning",
                recommendation="Add phone number for better contact options"
            ))
        elif not self.phone_pattern.match(personal_info.phone):
            report.add_issue(ValidationIssue(
                field="personal_info.phone",
                issue="Phone number format may not be standard",
                severity="info",
                recommendation="Use standard phone number format"
            ))
        
        if not personal_info.location:
            report.add_issue(ValidationIssue(
                field="personal_info.location",
                issue="Location/address is missing",
                severity="info",
                recommendation="Add city and state for location-based job matching"
            ))
    
    def _validate_summary(self, summary: Optional[str], report: ValidationReport):
        """Validate professional summary section."""
        if not summary:
            report.add_issue(ValidationIssue(
                field="summary",
                issue="Professional summary is missing",
                severity="warning",
                recommendation="Add 2-3 sentence professional summary"
            ))
        elif len(summary.strip()) < 50:
            report.add_issue(ValidationIssue(
                field="summary",
                issue="Professional summary is too short",
                severity="info",
                recommendation="Expand summary to 2-3 sentences highlighting key qualifications"
            ))
        elif len(summary.strip()) > 500:
            report.add_issue(ValidationIssue(
                field="summary",
                issue="Professional summary is too long",
                severity="info",
                recommendation="Condense summary to 2-3 impactful sentences"
            ))
    
    def _validate_work_experience(self, experience: List[WorkExperience], report: ValidationReport):
        """Validate work experience section."""
        if not experience:
            report.add_issue(ValidationIssue(
                field="experience",
                issue="No work experience found",
                severity="warning",
                recommendation="Add work experience or highlight relevant projects/internships"
            ))
            return
        
        for i, exp in enumerate(experience):
            if not exp.title:
                report.add_issue(ValidationIssue(
                    field=f"experience[{i}].title",
                    issue="Job title is missing",
                    severity="warning",
                    recommendation="Add job title for this position"
                ))
            
            if not exp.company:
                report.add_issue(ValidationIssue(
                    field=f"experience[{i}].company",
                    issue="Company name is missing",
                    severity="warning",
                    recommendation="Add company name for this position"
                ))
            
            if not exp.description or len(exp.description.strip()) < self.min_description_length:
                report.add_issue(ValidationIssue(
                    field=f"experience[{i}].description",
                    issue="Job description is missing or too short",
                    severity="warning",
                    recommendation="Add detailed description of responsibilities and achievements"
                ))
            
            if not exp.key_achievements:
                report.add_issue(ValidationIssue(
                    field=f"experience[{i}].key_achievements",
                    issue="No achievements listed for this position",
                    severity="info",
                    recommendation="Add specific achievements with quantified results"
                ))
    
    def _validate_education(self, education: List[Education], report: ValidationReport):
        """Validate education section."""
        if not education:
            report.add_issue(ValidationIssue(
                field="education",
                issue="No education information found",
                severity="info",
                recommendation="Add education background if applicable"
            ))
            return
        
        for i, edu in enumerate(education):
            if not edu.degree:
                report.add_issue(ValidationIssue(
                    field=f"education[{i}].degree",
                    issue="Degree type is missing",
                    severity="warning",
                    recommendation="Specify degree type (Bachelor's, Master's, etc.)"
                ))
            
            if not edu.field:
                report.add_issue(ValidationIssue(
                    field=f"education[{i}].field",
                    issue="Field of study is missing",
                    severity="info",
                    recommendation="Add field of study or major"
                ))
            
            if not edu.institution:
                report.add_issue(ValidationIssue(
                    field=f"education[{i}].institution",
                    issue="Institution name is missing",
                    severity="warning", 
                    recommendation="Add school/university name"
                ))
    
    def _validate_skills(self, skills, report: ValidationReport):
        """Validate skills section."""
        all_skills = skills.get_all_skills()
        
        if len(all_skills) == 0:
            report.add_issue(ValidationIssue(
                field="skills",
                issue="No skills listed",
                severity="critical",
                recommendation="Add relevant technical and professional skills"
            ))
        elif len(all_skills) < self.min_skills_count:
            report.add_issue(ValidationIssue(
                field="skills",
                issue=f"Only {len(all_skills)} skills listed (recommended: {self.min_skills_count}+)",
                severity="warning",
                recommendation="Add more relevant skills to improve job matching"
            ))
        
        # Check for technical skills balance
        technical_skills = skills.get_technical_skills()
        if len(technical_skills) == 0:
            report.add_issue(ValidationIssue(
                field="skills.technical",
                issue="No technical skills found",
                severity="info",
                recommendation="Add relevant technical skills if applicable"
            ))
    
    def _validate_projects(self, projects, report: ValidationReport):
        """Validate projects section."""
        if not projects:
            report.add_issue(ValidationIssue(
                field="projects",
                issue="No projects listed",
                severity="info",
                recommendation="Add relevant projects to demonstrate skills"
            ))
            return
        
        for i, project in enumerate(projects):
            if not project.technologies:
                report.add_issue(ValidationIssue(
                    field=f"projects[{i}].technologies",
                    issue="No technologies listed for project",
                    severity="info",
                    recommendation="Add technologies used in this project"
                ))
    
    def _validate_certifications(self, certifications, report: ValidationReport):
        """Validate certifications section."""
        # Certifications are optional, so only info-level suggestions
        if certifications:
            for i, cert in enumerate(certifications):
                if not cert.issuer:
                    report.add_issue(ValidationIssue(
                        field=f"certifications[{i}].issuer",
                        issue="Certification issuer is missing",
                        severity="info",
                        recommendation="Add issuing organization"
                    ))
    
    def _validate_job_basics(self, job: Job, report: ValidationReport):
        """Validate basic job information."""
        if not job.title:
            report.add_issue(ValidationIssue(
                field="job.title",
                issue="Job title is missing",
                severity="critical",
                recommendation="Job title is required for analysis"
            ))
        
        if not job.location:
            report.add_issue(ValidationIssue(
                field="job.location",
                issue="Job location is missing",
                severity="info",
                recommendation="Add job location for better matching"
            ))
    
    def _validate_job_requirements(self, requirements, report: ValidationReport):
        """Validate job requirements section."""
        if not requirements.required_skills and not requirements.preferred_skills:
            report.add_issue(ValidationIssue(
                field="job.requirements.skills",
                issue="No required or preferred skills specified",
                severity="warning",
                recommendation="Add required and preferred skills for better matching"
            ))
    
    def _validate_job_description(self, description: str, report: ValidationReport):
        """Validate job description."""
        if not description:
            report.add_issue(ValidationIssue(
                field="job.description",
                issue="Job description is missing",
                severity="critical",
                recommendation="Job description is required for ATS analysis"
            ))
        elif len(description.strip()) < 100:
            report.add_issue(ValidationIssue(
                field="job.description", 
                issue="Job description is too short",
                severity="warning",
                recommendation="Provide more detailed job description"
            ))
    
    def _validate_company_info(self, company, report: ValidationReport):
        """Validate company information."""
        if not company.name:
            report.add_issue(ValidationIssue(
                field="job.company.name",
                issue="Company name is missing",
                severity="critical",
                recommendation="Company name is required"
            ))
    
    def _calculate_resume_completeness(self, resume: Resume) -> float:
        """Calculate resume completeness percentage."""
        total_sections = 7
        completed_sections = 0
        
        if resume.personal_info.name and resume.personal_info.email:
            completed_sections += 1
        if resume.summary:
            completed_sections += 1
        if resume.experience:
            completed_sections += 1
        if resume.education:
            completed_sections += 1
        if resume.skills.get_all_skills():
            completed_sections += 1
        if resume.projects:
            completed_sections += 1
        if resume.certifications:
            completed_sections += 1
        
        return (completed_sections / total_sections) * 100
    
    def _calculate_job_completeness(self, job: Job) -> float:
        """Calculate job posting completeness percentage."""
        total_elements = 5
        completed_elements = 0
        
        if job.title:
            completed_elements += 1
        if job.description and len(job.description) > 50:
            completed_elements += 1
        if job.company.name:
            completed_elements += 1
        if job.requirements.required_skills or job.requirements.preferred_skills:
            completed_elements += 1
        if job.location:
            completed_elements += 1
        
        return (completed_elements / total_elements) * 100


# Global validator instance
ats_validator = ATSValidator()