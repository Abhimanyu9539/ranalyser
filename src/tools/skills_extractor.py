"""
LLM-based skills extraction system for resume analysis.
Uses OpenAI/LangChain with structured outputs for intelligent skill categorization.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass
from enum import Enum
import pprint

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field, validator

from src.models.resume import Skills, SkillCategory
from src.services.openai_service import langgraph_openai_service, OpenAIServiceError
from config.settings import settings
import dotenv
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillExtractionError(Exception):
    """Custom exception for skill extraction errors."""
    pass


class SkillConfidence(str, Enum):
    """Confidence levels for extracted skills."""
    HIGH = "high"        # Explicitly mentioned with context
    MEDIUM = "medium"    # Clearly implied or derived
    LOW = "low"         # Weakly implied or uncertain


class ExtractedSkill(BaseModel):
    """Individual skill with metadata."""
    name: str = Field(..., description="Standardized skill name")
    category: SkillCategory = Field(..., description="Skill category")
    confidence: SkillConfidence = Field(..., description="Extraction confidence")
    evidence: List[str] = Field(default_factory=list, description="Text evidence supporting this skill")
    proficiency_level: Optional[str] = Field(None, description="Proficiency if mentioned")
    years_experience: Optional[float] = Field(None, description="Years of experience if mentioned")
    context: Optional[str] = Field(None, description="Context where skill was found")
    
    @field_validator('name')
    def standardize_skill_name(cls, v):
        """Standardize skill names (e.g., JS -> JavaScript)."""
        skill_mappings = {
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'py': 'Python',
            'ml': 'Machine Learning',
            'ai': 'Artificial Intelligence',
            'nlp': 'Natural Language Processing',
            'cv': 'Computer Vision',
            'dl': 'Deep Learning',
            'aws': 'Amazon Web Services',
            'gcp': 'Google Cloud Platform',
            'k8s': 'Kubernetes',
            'docker': 'Docker',
            'react.js': 'React',
            'vue.js': 'Vue.js',
            'node.js': 'Node.js',
            'express.js': 'Express.js',
            'next.js': 'Next.js',
            'postgresql': 'PostgreSQL',
            'mongodb': 'MongoDB',
            'redis': 'Redis',
            'elasticsearch': 'Elasticsearch'
        }
        
        return skill_mappings.get(v.lower(), v.title())


class SkillExtractionResult(BaseModel):
    """Complete skill extraction result with metadata."""
    skills: List[ExtractedSkill] = Field(default_factory=list)
    categorized_skills: Skills = Field(default_factory=Skills)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_skills_by_category(self, category: SkillCategory) -> List[ExtractedSkill]:
        """Get all skills for a specific category."""
        return [skill for skill in self.skills if skill.category == category]
    
    def get_high_confidence_skills(self) -> List[ExtractedSkill]:
        """Get only high confidence skills."""
        return [skill for skill in self.skills if skill.confidence == SkillConfidence.HIGH]
    
    def get_skills_with_experience(self) -> List[ExtractedSkill]:
        """Get skills with mentioned years of experience."""
        return [skill for skill in self.skills if skill.years_experience is not None]


class LLMSkillExtractor:
    """LLM-based intelligent skill extraction system."""
    
    def __init__(self):
        """Initialize the skill extractor."""
        self.openai_service = langgraph_openai_service
        self.skill_knowledge_base = self._load_skill_knowledge()
        logger.info("LLM Skill Extractor initialized")
    
    def _load_skill_knowledge(self) -> Dict[str, List[str]]:
        """Load comprehensive skill knowledge base for better extraction."""
        return {
            "programming_languages": [
                "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
                "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB", "SQL",
                "HTML", "CSS", "Sass", "Less", "Shell", "Bash", "PowerShell"
            ],
            "frameworks_libraries": [
                "React", "Vue.js", "Angular", "Node.js", "Express.js", "Django", "Flask",
                "FastAPI", "Spring", "Laravel", "Ruby on Rails", "ASP.NET", "Next.js",
                "Nuxt.js", "Svelte", "jQuery", "Bootstrap", "Tailwind CSS", "Material-UI",
                "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Matplotlib"
            ],
            "databases": [
                "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
                "DynamoDB", "Oracle", "SQL Server", "SQLite", "Neo4j", "InfluxDB"
            ],
            "cloud_platforms": [
                "Amazon Web Services", "AWS", "Google Cloud Platform", "GCP", 
                "Microsoft Azure", "DigitalOcean", "Heroku", "Netlify", "Vercel"
            ],
            "tools_software": [
                "Docker", "Kubernetes", "Jenkins", "GitLab CI", "GitHub Actions",
                "Terraform", "Ansible", "Chef", "Puppet", "Vagrant", "Git", "SVN",
                "Jira", "Confluence", "Slack", "Figma", "Adobe Creative Suite",
                "Visual Studio Code", "IntelliJ IDEA", "PyCharm", "Sublime Text"
            ],
            "technical_skills": [
                "Machine Learning", "Deep Learning", "Natural Language Processing",
                "Computer Vision", "Data Science", "Data Analysis", "Statistical Analysis",
                "A/B Testing", "DevOps", "CI/CD", "Microservices", "API Development",
                "RESTful APIs", "GraphQL", "WebSocket", "Blockchain", "Cybersecurity",
                "Network Security", "Penetration Testing", "Agile", "Scrum", "Kanban"
            ],
            "soft_skills": [
                "Leadership", "Team Management", "Project Management", "Communication",
                "Problem Solving", "Critical Thinking", "Analytical Thinking",
                "Creativity", "Innovation", "Collaboration", "Mentoring", "Public Speaking",
                "Presentation", "Negotiation", "Time Management", "Adaptability",
                "Strategic Planning", "Decision Making", "Conflict Resolution"
            ]
        }
    
    async def extract_skills_from_text(
        self, 
        text: str, 
        context: str = "resume"
    ) -> SkillExtractionResult:
        """
        Extract skills from text using LLM with intelligent categorization.
        
        Args:
            text: The text to extract skills from
            context: Context of the text (resume, job_description, etc.)
        
        Returns:
            SkillExtractionResult with categorized skills and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting LLM skill extraction from {context}")
            
            # Create the extraction prompt
            extraction_result = await self._perform_llm_extraction(text, context)
            
            # Post-process and validate skills
            processed_skills = await self._post_process_skills(extraction_result.skills, text)

            # Create categorized skills structure
            categorized_skills = self._categorize_skills(processed_skills)

            # Calculate metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time": processing_time,
                "total_skills_found": len(processed_skills),
                "high_confidence_count": len([s for s in processed_skills if s.confidence == SkillConfidence.HIGH]),
                "categories_found": len([cat for cat in SkillCategory if len(categorized_skills.__dict__[cat.value]) > 0]),
                "context": context,
                "model_used": settings.openai_model
            }
            
            result = SkillExtractionResult(
                skills=processed_skills,
                categorized_skills=categorized_skills,
                extraction_metadata=metadata
            )
            
            logger.info(f"Skill extraction completed. Found {len(processed_skills)} skills in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Skill extraction failed: {e}")
            raise SkillExtractionError(f"Failed to extract skills: {e}")
    
    async def _perform_llm_extraction(self, text: str, context: str) -> SkillExtractionResult:
        """Perform context-aware LLM-based skill extraction."""
        
        # Create wrapper model for list of skills
        class SkillExtractionLLMResult(BaseModel):
            skills: List[ExtractedSkill]
        
        structured_llm = self.openai_service.llm.with_structured_output(SkillExtractionLLMResult)
        
        # Base system prompt
        base_system_prompt = """You are an expert skill extraction specialist with deep knowledge of technology, business, 
        and professional skills.

        EXTRACTION PRINCIPLES:
        1. Extract both explicitly mentioned and reasonably implied skills
        2. Standardize skill names (e.g., "JS" → "JavaScript", "ML" → "Machine Learning")
        3. Categorize skills accurately using the provided categories
        4. Assess confidence based on how clearly the skill is demonstrated
        5. Look for proficiency levels and years of experience when mentioned
        6. Provide specific text evidence for each extracted skill

        CONFIDENCE LEVELS:
        - HIGH: Explicitly mentioned with clear context or demonstrated usage
        - MEDIUM: Clearly implied from context or experience descriptions
        - LOW: Weakly implied or uncertain

        SKILL CATEGORIES:
        - technical_skills: General technical concepts, methodologies, domains
        - programming_languages: Programming and scripting languages
        - frameworks_libraries: Software frameworks, libraries, SDKs
        - tools_software: Development tools, software applications
        - databases: Database systems and query languages  
        - cloud_platforms: Cloud services and platforms
        - soft_skills: Leadership, communication, interpersonal skills
        - domain_expertise: Industry-specific knowledge and business domains"""

        # Context-specific instructions
        context_instructions = self._get_context_specific_instructions(context)
        
        # Combine base prompt with context-specific instructions
        system_prompt = f"{base_system_prompt}\n\n{context_instructions}"
        
        # Context-specific human prompt
        human_prompt = self._get_context_specific_human_prompt(context)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        
        chain = prompt | structured_llm
        
        result = await chain.ainvoke({
            "text": text,
            "context": context
        })
        
        return SkillExtractionResult(skills=result.skills)

    def _get_context_specific_instructions(self, context: str) -> str:
        """Get context-specific extraction instructions."""
        
        if context == "resume":
            return """
            RESUME-SPECIFIC INSTRUCTIONS:
            - Focus on DEMONSTRATED skills and experience
            - Look for skills in work experience, projects, and achievements
            - Pay attention to leadership roles and team management
            - Extract years of experience when mentioned
            - Consider education and certifications as skill indicators
            - Infer technical skills from project descriptions
            - Weight skills based on how extensively they're described
            - Look for progression and growth in skill levels over time

            WHAT TO EXTRACT FROM RESUMES:
            - Skills explicitly listed in skills sections
            - Technologies mentioned in job descriptions
            - Programming languages and frameworks used in projects
            - Tools and platforms mentioned in work experience
            - Soft skills demonstrated through achievements (led team = leadership)
            - Domain expertise from industry experience
            - Certifications and their implied skills

            CONFIDENCE ASSESSMENT FOR RESUMES:
            - HIGH: Skills with specific examples, years of experience, or major projects
            - MEDIUM: Skills mentioned in job duties or project lists
            - LOW: Skills only briefly mentioned or weakly implied"""

        elif context == "job_description":
            return """
            JOB-SPECIFIC INSTRUCTIONS:
            - Distinguish between REQUIRED vs PREFERRED skills
            - Extract both hard requirements and nice-to-have skills
            - Look for minimum experience requirements
            - Identify industry-specific skills and knowledge
            - Pay attention to seniority level indicators
            - Extract team size and management expectations
            - Look for specific tools, technologies, and methodologies

            WHAT TO EXTRACT FROM JOB DESCRIPTIONS:
            - Must-have technical skills and qualifications
            - Preferred additional skills and experience
            - Software, tools, and platforms mentioned
            - Years of experience requirements
            - Education and certification requirements
            - Industry knowledge and domain expertise
            - Soft skills and interpersonal requirements
            - Management and leadership expectations

            CONFIDENCE ASSESSMENT FOR JOB DESCRIPTIONS:
            - HIGH: Skills marked as "required", "must have", or with specific experience years
            - MEDIUM: Skills in "preferred" or "nice to have" sections
            - LOW: Skills only briefly mentioned or context-dependent"""

        elif context == "project_description":
            return """
            PROJECT-SPECIFIC INSTRUCTIONS:
            - Focus on the TECHNICAL STACK and implementation details
            - Extract development methodologies and practices
            - Look for specific technologies, frameworks, and tools used
            - Identify problem-solving approaches and techniques
            - Pay attention to architectural patterns and design choices
            - Extract integration and deployment technologies

            WHAT TO EXTRACT FROM PROJECT DESCRIPTIONS:
            - Programming languages and frameworks used
            - Databases and data storage solutions
            - Cloud services and deployment platforms
            - Development tools and IDEs mentioned
            - Testing frameworks and methodologies
            - DevOps and CI/CD tools
            - APIs and integration technologies
            - Frontend and backend technologies

            CONFIDENCE ASSESSMENT FOR PROJECTS:
            - HIGH: Technologies explicitly mentioned as being used/implemented
            - MEDIUM: Technologies implied by architecture or requirements
            - LOW: Technologies mentioned peripherally or in dependencies"""

        else:  # generic context
            return """
            GENERAL TEXT INSTRUCTIONS:
            - Extract any skills that can be reasonably identified
            - Use context clues to determine skill categories
            - Be conservative with confidence levels for unclear text
            - Look for both technical and soft skills
            - Consider the overall tone and style of the text"""

    def _get_context_specific_human_prompt(self, context: str) -> str:
        """Get context-specific human prompts."""
        
        base_prompt = "Extract all relevant skills from this {context} text:\n\nTEXT:\n{text}\n\n"
        
        if context == "resume":
            specific_instructions = """
            RESUME ANALYSIS FOCUS:
            Analyze this resume to extract skills that demonstrate the candidate's capabilities:

            1. **Work Experience Skills**: What technologies, tools, and methodologies did they use?
            2. **Leadership & Management**: Any evidence of team leadership, project management, or mentoring?
            3. **Technical Progression**: How have their technical skills evolved over time?
            4. **Domain Expertise**: What industries or business areas do they have experience in?
            5. **Quantified Experience**: Look for specific years of experience with technologies
            6. **Achievement-Based Skills**: Skills demonstrated through specific accomplishments

            Focus on skills that would be valuable to potential employers and provide strong evidence for each skill extracted.
            """

        elif context == "job_description":
            specific_instructions = """
            JOB REQUIREMENTS ANALYSIS:
            Analyze this job description to extract skill requirements for potential candidates:

            1. **Must-Have Skills**: What are the non-negotiable requirements?
            2. **Preferred Skills**: What would make a candidate stand out?
            3. **Experience Requirements**: What minimum years of experience are needed?
            4. **Team & Leadership**: What management or collaboration skills are needed?
            5. **Industry Knowledge**: What domain expertise is required?
            6. **Growth Potential**: What skills might be developed in this role?

            Categorize skills by importance level and provide clear evidence from the job posting.
            """

        elif context == "project_description":
            specific_instructions = """
            PROJECT TECHNOLOGY ANALYSIS:
            Analyze this project description to extract the technical skills and technologies involved:

            1. **Core Technologies**: What programming languages and frameworks are used?
            2. **Infrastructure**: What deployment, hosting, and scaling technologies?
            3. **Data & Storage**: What databases and data processing tools?
            4. **Development Process**: What methodologies, testing, and collaboration tools?
            5. **Integration**: What APIs, services, and third-party tools?
            6. **Specialized Skills**: Any domain-specific or advanced technical skills?

            Focus on extracting the complete technical stack and development practices.
            """

        else:
            specific_instructions = """
            GENERAL SKILL ANALYSIS:
            Extract any identifiable skills from this text, paying attention to:

            1. **Technical Skills**: Any technology, tools, or methodologies mentioned
            2. **Professional Skills**: Leadership, communication, and work-related abilities  
            3. **Context Clues**: Infer skills from activities and responsibilities described
            4. **Industry Terms**: Domain-specific knowledge and expertise

            Be thorough but conservative with confidence levels given the general context.
            """

        return base_prompt + specific_instructions
    
    async def _post_process_skills(self, skills: List[ExtractedSkill], original_text: str) -> List[ExtractedSkill]:
        """Post-process extracted skills for deduplication and validation."""
        
        # Deduplicate skills by standardized name and category
        unique_skills = {}
        
        for skill in skills:
            key = (skill.name.lower(), skill.category)
            
            if key not in unique_skills:
                unique_skills[key] = skill
            else:
                # Merge skills with same name/category, keeping higher confidence
                existing = unique_skills[key]
                if skill.confidence.value > existing.confidence.value:
                    # Merge evidence
                    skill.evidence.extend(existing.evidence)
                    unique_skills[key] = skill
                else:
                    existing.evidence.extend(skill.evidence)
        
        # Convert back to list and sort by confidence
        processed_skills = list(unique_skills.values())
        processed_skills.sort(key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}[x.confidence.value],
            x.name
        ), reverse=True)
        
        return processed_skills
    
    def _categorize_skills(self, skills: List[ExtractedSkill]) -> Skills:
        """Convert extracted skills to categorized Skills model."""
        categorized = Skills()
        
        for skill in skills:
            skill_name = skill.name
            category = skill.category
            
            # Get the appropriate list from Skills model
            if category == SkillCategory.TECHNICAL:
                categorized.technical_skills.append(skill_name)
            elif category == SkillCategory.PROGRAMMING:
                categorized.programming_languages.append(skill_name)
            elif category == SkillCategory.FRAMEWORKS:
                categorized.frameworks_libraries.append(skill_name)
            elif category == SkillCategory.TOOLS:
                categorized.tools_software.append(skill_name)
            elif category == SkillCategory.DATABASES:
                categorized.databases.append(skill_name)
            elif category == SkillCategory.CLOUD:
                categorized.cloud_platforms.append(skill_name)
            elif category == SkillCategory.SOFT_SKILLS:
                categorized.soft_skills.append(skill_name)
            elif category == SkillCategory.DOMAIN:
                categorized.domain_expertise.append(skill_name)
        
        return categorized
    
    async def extract_skills_from_resume(self, resume_text: str) -> SkillExtractionResult:
        """Extract skills specifically from resume text."""
        return await self.extract_skills_from_text(resume_text, "resume")
    
    async def extract_skills_from_job_description(self, job_text: str) -> SkillExtractionResult:
        """Extract skills specifically from job description text."""
        return await self.extract_skills_from_text(job_text, "job_description")
    
    async def compare_skill_sets(
        self, 
        resume_skills: SkillExtractionResult, 
        job_skills: SkillExtractionResult
    ) -> Dict[str, Any]:
        """Compare skills between resume and job requirements."""
        
        resume_skill_names = {skill.name.lower() for skill in resume_skills.skills}
        job_skill_names = {skill.name.lower() for skill in job_skills.skills}
        
        matched_skills = resume_skill_names.intersection(job_skill_names)
        missing_skills = job_skill_names - resume_skill_names
        additional_skills = resume_skill_names - job_skill_names
        
        # Calculate match percentage
        total_job_skills = len(job_skill_names)
        match_percentage = (len(matched_skills) / total_job_skills * 100) if total_job_skills > 0 else 0
        
        return {
            "match_percentage": round(match_percentage, 1),
            "matched_skills": sorted(list(matched_skills)),
            "missing_skills": sorted(list(missing_skills)),
            "additional_skills": sorted(list(additional_skills)),
            "total_resume_skills": len(resume_skill_names),
            "total_job_skills": len(job_skill_names),
            "matched_count": len(matched_skills)
        }
    
    def get_skill_suggestions(self, extracted_skills: SkillExtractionResult) -> Dict[str, List[str]]:
        """Suggest related skills based on extracted skills."""
        suggestions = {}
        
        skill_relationships = {
            "Python": ["Django", "Flask", "FastAPI", "Pandas", "NumPy", "Scikit-learn"],
            "JavaScript": ["React", "Node.js", "TypeScript", "Vue.js", "Angular"],
            "React": ["Redux", "Next.js", "TypeScript", "Jest", "Material-UI"],
            "AWS": ["Docker", "Kubernetes", "Terraform", "Jenkins", "Linux"],
            "Machine Learning": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "Data Science"],
            "Docker": ["Kubernetes", "Jenkins", "AWS", "Linux", "DevOps"]
        }
        
        current_skills = {skill.name for skill in extracted_skills.skills}
        
        for skill in current_skills:
            if skill in skill_relationships:
                related = [s for s in skill_relationships[skill] if s not in current_skills]
                if related:
                    suggestions[skill] = related[:3]  # Top 3 suggestions
        
        return suggestions


# Global extractor instance
llm_skill_extractor = LLMSkillExtractor()


# Convenience functions
async def extract_skills_from_resume(resume_text: str) -> SkillExtractionResult:
    """Extract skills from resume text."""
    return await llm_skill_extractor.extract_skills_from_resume(resume_text)


async def extract_skills_from_job(job_text: str) -> SkillExtractionResult:
    """Extract skills from job description."""
    return await llm_skill_extractor.extract_skills_from_job_description(job_text)


async def compare_resume_job_skills(resume_text: str, job_text: str) -> Dict[str, Any]:
    """Compare skills between resume and job description."""
    resume_skills = await extract_skills_from_resume(resume_text)
    job_skills = await extract_skills_from_job(job_text)
    
    comparison = await llm_skill_extractor.compare_skill_sets(resume_skills, job_skills)
    
    return {
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "comparison": comparison
    }


def get_skill_suggestions(skills_result: SkillExtractionResult) -> Dict[str, List[str]]:
    """Get skill suggestions based on current skills."""
    return llm_skill_extractor.get_skill_suggestions(skills_result)


if __name__ == "__main__":
    with open("/home/user/ranalyser/tests/data/sample_resumes/senior_developer.txt", "r") as file:
        resume_text = file.read()
    
    with open("/home/user/ranalyser/tests/data/sample_jobs/frontend_engineer.txt", "r") as f:
        job_text = f.read()

    
    #result = asyncio.run(extract_skills_from_resume(resume_text))
    #print(result)
    result = asyncio.run(compare_resume_job_skills(resume_text=resume_text, job_text=job_text))
    print(result)
