"""
LLM-based projects extraction from resume text.
Uses OpenAI/LangChain with function calling for reliable extraction of project information.
Extracts personal projects, work projects, and academic projects with technical details.
"""
import logging
import time
import re
from typing import Dict, List, Optional, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from src.models.resume import Project
from src.services.openai_service import langgraph_openai_service, OpenAIServiceError
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectsExtractionError(Exception):
    """Custom exception for projects extraction errors."""
    pass


class ProjectsExtractionResult(BaseModel):
    """Projects extraction result with metadata."""
    projects: List[Project] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_recent_projects(self, years: int = 2) -> List[Project]:
        """Get projects from the last N years."""
        from datetime import datetime
        current_year = datetime.now().year
        cutoff_year = current_year - years
        
        recent = []
        for project in self.projects:
            if project.end_date:
                try:
                    # Extract year from end date
                    year_match = re.search(r'\b(20\d{2})\b', project.end_date)
                    if year_match and int(year_match.group()) >= cutoff_year:
                        recent.append(project)
                except:
                    continue
            elif project.start_date:
                try:
                    # If no end date, check start date
                    year_match = re.search(r'\b(20\d{2})\b', project.start_date)
                    if year_match and int(year_match.group()) >= cutoff_year:
                        recent.append(project)
                except:
                    continue
        return recent
    
    def get_projects_by_technology(self, technology: str) -> List[Project]:
        """Get projects that used a specific technology."""
        tech_lower = technology.lower()
        matching_projects = []
        for project in self.projects:
            # Check in technologies list
            for tech in project.technologies:
                if tech_lower in tech.lower():
                    matching_projects.append(project)
                    break
            # Also check in description and key features
            if tech_lower in project.description.lower():
                if project not in matching_projects:
                    matching_projects.append(project)
        return matching_projects
    
    def get_web_projects(self) -> List[Project]:
        """Get web development projects."""
        web_keywords = ['web', 'website', 'html', 'css', 'javascript', 'react', 'vue', 'angular', 'node', 'express', 'django', 'flask']
        web_projects = []
        for project in self.projects:
            project_text = f"{project.name} {project.description} {' '.join(project.technologies)}".lower()
            if any(keyword in project_text for keyword in web_keywords):
                web_projects.append(project)
        return web_projects
    
    def get_mobile_projects(self) -> List[Project]:
        """Get mobile development projects."""
        mobile_keywords = ['mobile', 'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin', 'app store', 'play store']
        mobile_projects = []
        for project in self.projects:
            project_text = f"{project.name} {project.description} {' '.join(project.technologies)}".lower()
            if any(keyword in project_text for keyword in mobile_keywords):
                mobile_projects.append(project)
        return mobile_projects
    
    def get_projects_with_links(self) -> List[Project]:
        """Get projects that have URLs (live demos, GitHub, etc.)."""
        return [project for project in self.projects if project.url]
    
    def get_technologies_used(self) -> List[str]:
        """Get all unique technologies used across projects."""
        all_technologies = []
        for project in self.projects:
            all_technologies.extend(project.technologies)
        return list(set(all_technologies))


class ProjectsLLMResult(BaseModel):
    """Wrapper for LLM extraction result."""
    projects: List[Project] = Field(default_factory=list)


class LLMProjectsExtractor:
    """LLM-based projects extraction system."""
    
    def __init__(self):
        """Initialize the projects extractor."""
        self.openai_service = langgraph_openai_service
        logger.info("LLM Projects Extractor initialized")
    
    async def extract_projects_from_text(self, text: str) -> ProjectsExtractionResult:
        """
        Extract projects from resume text using LLM.
        
        Args:
            text: The resume text to extract projects from
        
        Returns:
            ProjectsExtractionResult with extracted projects data and metadata
        """
        start_time = time.time()
        
        try:
            logger.info("Starting LLM projects extraction")
            
            # Perform LLM extraction
            projects_list = await self._perform_llm_extraction(text)
            
            # Post-process projects
            processed_projects = self._post_process_projects(projects_list)
            
            # Calculate metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time": processing_time,
                "total_projects": len(processed_projects),
                "projects_with_urls": len([p for p in processed_projects if p.url]),
                "unique_technologies": len(self._get_unique_technologies(processed_projects)),
                "web_projects": self._get_web_projects_count(processed_projects),
                "mobile_projects": self._get_mobile_projects_count(processed_projects),
                "model_used": settings.openai_model,
                "extraction_method": "llm_function_calling"
            }
            
            result = ProjectsExtractionResult(
                projects=processed_projects,
                extraction_metadata=metadata
            )
            
            logger.info(f"Projects extraction completed. Found {len(processed_projects)} projects in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Projects extraction failed: {e}")
            raise ProjectsExtractionError(f"Failed to extract projects: {e}")
    
    async def _perform_llm_extraction(self, text: str) -> List[Project]:
        """Perform the actual LLM-based projects extraction."""
        
        # Use function calling for reliable extraction
        structured_llm = self.openai_service.llm.with_structured_output(
            ProjectsLLMResult,
            method="function_calling"
        )
        
        # Create specialized projects extraction prompt
        system_prompt = """You are an expert at extracting project information from resumes.
        Extract ALL types of projects including personal projects, work projects, academic projects, and side projects.

        EXTRACTION GUIDELINES:
        1. Extract each project as a separate entry
        2. Include personal projects, work projects, academic projects, hackathons, contributions
        3. Extract technical details, technologies used, and key features
        4. Preserve original project names and descriptions
        5. Extract URLs for live demos, GitHub repos, or project sites
        6. Include project duration/timeline when mentioned
        7. Focus on technical implementation and impact

        WHAT TO EXTRACT FOR EACH PROJECT:
        - Project name (exact title/name)
        - Project description (what it does, purpose, functionality)
        - Technologies used (programming languages, frameworks, tools, databases)
        - Project URL (GitHub, live demo, portfolio link, etc.)
        - Start date (when project began, if mentioned)
        - End date (when completed, if mentioned, or "Ongoing" for active projects)
        - Key features (main functionality, notable achievements)

        PROJECT TYPES TO INCLUDE:
        - Personal/Side Projects: Individual coding projects, apps, websites
        - Work Projects: Professional projects from jobs/internships
        - Academic Projects: School assignments, capstone projects, research
        - Open Source: Contributions to open source projects
        - Hackathon Projects: Competition entries and hackathon builds
        - Freelance Projects: Client work and consulting projects

        TECHNICAL FOCUS:
        - Programming languages used (Python, JavaScript, Java, etc.)
        - Frameworks and libraries (React, Django, Express, etc.)
        - Databases and storage (PostgreSQL, MongoDB, Redis, etc.)
        - Cloud services and deployment (AWS, Heroku, Netlify, etc.)
        - Development tools and practices (Git, Docker, CI/CD, etc.)

        DATE EXTRACTION RULES:
        - Extract project dates exactly as shown in the resume
        - Common formats: "2023", "Jan 2023 - Mar 2023", "2022 - Present"
        - Use "Ongoing" or "Present" for active projects
        - Use "Not specified" if no dates mentioned

        URL EXTRACTION:
        - GitHub repositories: github.com/username/project
        - Live demos: project domain or hosting URL
        - Portfolio links: personal website project pages
        - Extract the full URL exactly as shown

        KEY FEATURES EXTRACTION:
        - Main functionality and capabilities
        - Notable technical achievements
        - User metrics or impact when mentioned
        - Special features or innovations
        - Performance improvements or optimizations"""

        human_prompt = """Extract all project information from this resume text:

        RESUME TEXT:
        {text}

        Look for projects in sections like:
        - "Projects" / "Personal Projects" / "Side Projects"
        - "Work Projects" / "Professional Projects"
        - "Academic Projects" / "School Projects"
        - "Portfolio" / "Notable Work"
        - "Open Source Contributions"
        - "Hackathon Projects"

        For each PROJECT found, extract:
        1. **Project Identity**: Name and clear description of what it does
        2. **Technical Details**: All technologies, languages, frameworks used
        3. **Implementation**: How it was built, architecture, key technical decisions
        4. **Features**: Main functionality, user-facing features, capabilities
        5. **Links**: GitHub repos, live demos, project URLs
        6. **Timeline**: When it was built, duration, current status

        IMPORTANT: Focus on technical projects that demonstrate programming/development skills:
        ✅ INCLUDE:
        - Web applications and websites
        - Mobile apps (iOS, Android, React Native, etc.)
        - Desktop applications and software tools
        - APIs and backend services
        - Data analysis and machine learning projects
        - DevOps and infrastructure projects
        - Open source contributions
        - Technical research projects

        ❌ DO NOT INCLUDE:
        - Non-technical projects (marketing campaigns, business plans)
        - Work experience (this should be in work history, not projects)
        - Educational degrees or courses
        - Certifications or training programs

        Extract dates exactly as written in the resume. Do not modify the format.

        If any information is missing or unclear, use "Not specified" for that field.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"text": text})
        
        return result.projects
    
    def _post_process_projects(self, projects_list: List[Project]) -> List[Project]:
        """Post-process and validate extracted projects."""
        
        processed = []
        
        for project in projects_list:
            try:
                # Clean and validate the project entry
                cleaned_project = self._clean_project_entry(project)
                
                # Skip entries with missing critical information
                if not cleaned_project.name or not cleaned_project.description:
                    logger.warning(f"Skipping incomplete project: {cleaned_project.name}")
                    continue
                
                # Skip very short or unclear descriptions
                if len(cleaned_project.description.strip()) < 20:
                    logger.warning(f"Skipping project with short description: {cleaned_project.name}")
                    continue
                
                # Skip if no technologies mentioned (likely not a technical project)
                if not cleaned_project.technologies:
                    logger.warning(f"Skipping non-technical project: {cleaned_project.name}")
                    continue
                
                processed.append(cleaned_project)
                
            except Exception as e:
                logger.warning(f"Error processing project entry: {e}")
                continue
        
        # Sort by date (most recent first, ongoing projects first)
        processed.sort(
            key=lambda x: (
                x.end_date != "Ongoing" and x.end_date != "Present",  # Ongoing first
                self._parse_project_date_for_sorting(x.end_date or x.start_date)
            ),
            reverse=True
        )
        
        return processed
    
    def _clean_project_entry(self, project: Project) -> Project:
        """Clean and standardize a single project entry."""
        
        # Clean project name
        name = project.name.strip() if project.name else ""
        name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
        
        # Clean description
        description = project.description.strip() if project.description else ""
        description = re.sub(r'\s+', ' ', description)  # Normalize whitespace
        description = re.sub(r'^[•\-\*]\s*', '', description)  # Remove leading bullets
        
        # Clean and deduplicate technologies
        technologies = []
        if project.technologies:
            seen = set()
            for tech in project.technologies:
                clean_tech = tech.strip()
                clean_tech = self._standardize_technology_name(clean_tech)
                if clean_tech and clean_tech.lower() not in seen:
                    technologies.append(clean_tech)
                    seen.add(clean_tech.lower())
        
        # Clean URL
        url = str(project.url).strip() if project.url else None
        if url:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            # Clean common URL issues
            if url.lower() in ['not specified', 'n/a', 'na', 'none']:
                url = None
        
        # Clean dates (preserve format)
        start_date = self._clean_project_date(project.start_date)
        end_date = self._clean_project_date(project.end_date)
        
        # Clean and deduplicate key features
        key_features = []
        if project.key_features:
            seen = set()
            for feature in project.key_features:
                clean_feature = feature.strip()
                clean_feature = re.sub(r'^[•\-\*]\s*', '', clean_feature)  # Remove bullets
                if clean_feature and clean_feature.lower() not in seen:
                    key_features.append(clean_feature)
                    seen.add(clean_feature.lower())
        
        return Project(
            name=name,
            description=description,
            technologies=technologies,
            url=url,
            start_date=start_date,
            end_date=end_date,
            key_features=key_features
        )
    
    def _standardize_technology_name(self, tech: str) -> str:
        """Standardize technology names for consistency."""
        if not tech:
            return tech
        
        # Common technology name standardizations
        standardizations = {
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'py': 'Python',
            'reactjs': 'React',
            'react.js': 'React',
            'vuejs': 'Vue.js',
            'vue.js': 'Vue.js',
            'nodejs': 'Node.js',
            'node.js': 'Node.js',
            'expressjs': 'Express.js',
            'express.js': 'Express.js',
            'nextjs': 'Next.js',
            'next.js': 'Next.js',
            'postgresql': 'PostgreSQL',
            'postgres': 'PostgreSQL',
            'mongodb': 'MongoDB',
            'mongo': 'MongoDB',
            'mysql': 'MySQL',
            'sqlite': 'SQLite',
            'redis': 'Redis',
            'aws': 'Amazon Web Services',
            'gcp': 'Google Cloud Platform',
            'docker': 'Docker',
            'kubernetes': 'Kubernetes',
            'k8s': 'Kubernetes'
        }
        
        tech_lower = tech.lower()
        for key, standard in standardizations.items():
            if tech_lower == key:
                return standard
        
        # Return title case if no standardization found
        return tech.title()
    
    def _clean_project_date(self, date_str: Optional[str]) -> Optional[str]:
        """Clean project date while preserving format."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Handle common variations
        if date_str.lower() in ['not specified', 'n/a', 'na', 'none', 'tbd']:
            return None
        
        # Handle ongoing/present variations
        if date_str.lower() in ['ongoing', 'present', 'current', 'in progress', 'active']:
            return "Ongoing"
        
        return date_str
    
    def _parse_project_date_for_sorting(self, date_str: Optional[str]) -> int:
        """Parse project date for sorting purposes."""
        if not date_str:
            return 0
        
        if date_str.lower() in ['ongoing', 'present']:
            return 9999  # Sort ongoing projects first
        
        # Extract year from various formats
        try:
            # Look for 4-digit year
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                return int(year_match.group())
            
            # Try direct conversion
            return int(date_str)
        except (ValueError, TypeError):
            return 0
    
    def _get_unique_technologies(self, projects: List[Project]) -> List[str]:
        """Helper method to get unique technologies for metadata."""
        result = ProjectsExtractionResult(projects=projects)
        return result.get_technologies_used()
    
    def _get_web_projects_count(self, projects: List[Project]) -> int:
        """Helper method to count web projects for metadata."""
        result = ProjectsExtractionResult(projects=projects)
        return len(result.get_web_projects())
    
    def _get_mobile_projects_count(self, projects: List[Project]) -> int:
        """Helper method to count mobile projects for metadata."""
        result = ProjectsExtractionResult(projects=projects)
        return len(result.get_mobile_projects())


# Global extractor instance
llm_projects_extractor = LLMProjectsExtractor()


# Convenience functions
async def extract_projects_from_resume(resume_text: str) -> ProjectsExtractionResult:
    """Extract projects from resume text."""
    return await llm_projects_extractor.extract_projects_from_text(resume_text)


async def extract_projects_list(resume_text: str) -> List[Project]:
    """Extract projects and return just the list of Project objects."""
    result = await extract_projects_from_resume(resume_text)
    return result.projects


if __name__ == "__main__":
    # Test with sample data
    sample_resume = """
    John Doe
    Senior Software Engineer
    
    PROJECTS:
    
    E-Commerce Platform | Personal Project | 2023 - Present
    Full-stack e-commerce web application built with React and Node.js
    • Built responsive frontend with React, Redux, and Material-UI
    • Developed RESTful API with Node.js, Express, and MongoDB
    • Implemented user authentication with JWT and bcrypt
    • Integrated Stripe payment processing and email notifications
    • Deployed on AWS EC2 with Nginx reverse proxy
    Technologies: React, Node.js, Express, MongoDB, JWT, Stripe, AWS, Docker
    GitHub: https://github.com/johndoe/ecommerce-platform
    Live Demo: https://ecommerce-demo.johndoe.com
    
    Stock Price Predictor | Machine Learning Project | Jan 2023 - Mar 2023
    Machine learning model to predict stock prices using historical data
    • Collected and processed stock data using Yahoo Finance API
    • Built LSTM neural network with TensorFlow and Keras
    • Achieved 85% prediction accuracy on test dataset
    • Created interactive dashboard with Streamlit for visualization
    • Deployed model as REST API using Flask
    Technologies: Python, TensorFlow, Keras, Pandas, NumPy, Streamlit, Flask
    GitHub: https://github.com/johndoe/stock-predictor
    
    Mobile Task Manager | React Native App | 2022
    Cross-platform mobile app for task and project management
    • Built with React Native and TypeScript for iOS and Android
    • Implemented offline-first architecture with SQLite local storage
    • Added push notifications using Firebase Cloud Messaging
    • Designed intuitive UI with gesture-based interactions
    • Published on both App Store and Google Play Store
    Technologies: React Native, TypeScript, SQLite, Firebase, Redux
    App Store: https://apps.apple.com/app/taskmanager
    """
    
    print("Testing ProjectsExtractor...")
    try:
        import asyncio
        result = asyncio.run(extract_projects_from_resume(sample_resume))
        
        print("✅ Extraction successful!")
        print(f"Found {len(result.projects)} projects")
        print(f"Technologies used: {result.get_technologies_used()[:8]}")  # Show first 8
        print(f"Web projects: {len(result.get_web_projects())}")
        print(f"Mobile projects: {len(result.get_mobile_projects())}")
        print(f"Projects with URLs: {len(result.get_projects_with_links())}")
        print(f"Processing time: {result.extraction_metadata['processing_time']:.2f}s")
        
        print("\nExtracted projects:")
        for i, project in enumerate(result.projects, 1):
            print(f"{i}. {project.name}")
            print(f"   Timeline: {project.start_date} - {project.end_date}")
            print(f"   Technologies: {', '.join(project.technologies[:5])}")  # Show first 5
            if project.url:
                print(f"   URL: {project.url}")
            print(f"   Features: {len(project.key_features)} key features")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with real file if available
    try:
        with open("/home/user/ranalyser/tests/data/sample_resumes/senior_developer.txt", "r") as file:
            resume_text = file.read()
        
        print("=" * 60)
        print("Testing with real resume file...")
        result = asyncio.run(extract_projects_from_resume(resume_text))
        
        print("✅ Real file test successful!")
        print(f"Found {len(result.projects)} projects")
        print(f"Processing time: {result.extraction_metadata['processing_time']:.2f}s")
        
        for i, project in enumerate(result.projects, 1):
            print(f"{i}. {project.name}")
            print(f"   Timeline: {project.start_date} - {project.end_date}")
            print(f"   Technologies: {', '.join(project.technologies[:3])}")
            
    except FileNotFoundError:
        print("Sample resume file not found, skipping file test")
    except Exception as e:
        print(f"Real file test error: {e}")