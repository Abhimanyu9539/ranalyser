"""
Updated prompt templates for structured outputs using Pydantic models.
Removed JSON formatting instructions as they're handled by with_structured_output().
"""

class ResumePrompts:
    """Prompts for resume parsing and analysis."""
    
    PARSE_RESUME = """
    You are an expert resume parser. Extract structured information from the following resume text.
    
    Resume Text:
    {resume_text}
    
    Extract the following information accurately:
    
    PERSONAL INFORMATION:
    - Full name, email, phone number
    - Location (city, state/country)
    - LinkedIn profile URL, portfolio/website URL, GitHub URL
    
    PROFESSIONAL SUMMARY:
    - Professional summary or objective statement
    
    WORK EXPERIENCE:
    - Job title, company name, location
    - Start and end dates (format as MM/YYYY or "Present")
    - Job description and key achievements
    - Technologies or tools used
    
    EDUCATION:
    - Degree type and field of study
    - Institution name
    - Graduation date (MM/YYYY format)
    - GPA if mentioned, honors/awards if mentioned
    
    SKILLS:
    - Technical skills (software, platforms, methodologies)
    - Programming languages
    - Frameworks and libraries
    - Tools and software
    - Databases
    - Cloud platforms
    - Soft skills (leadership, communication, etc.)
    - Domain expertise (industry-specific knowledge)
    
    CERTIFICATIONS:
    - Certification name and issuing organization
    - Date obtained and expiry date
    - Credential ID if available
    
    PROJECTS:
    - Project name and description
    - Technologies used
    - Project URL if available
    - Key features or achievements
    
    INSTRUCTIONS:
    1. Extract only information that is explicitly mentioned in the resume
    2. Use "Not specified" for missing optional fields
    3. Standardize date formats to MM/YYYY
    4. Clean and normalize company names, job titles, and skills
    5. Group similar skills together logically
    6. Do not make up or infer information that isn't clearly stated
    """
    
    EXTRACT_SKILLS = """
    You are a skills extraction expert. Analyze the following resume text and extract all relevant skills.
    
    Resume Text:
    {resume_text}
    
    Categorize skills into these specific categories:
    
    TECHNICAL SKILLS: Software, platforms, methodologies, tools (non-programming)
    PROGRAMMING LANGUAGES: All programming/scripting languages
    FRAMEWORKS & LIBRARIES: Web frameworks, libraries, SDKs
    TOOLS & SOFTWARE: Development tools, IDEs, design software, productivity tools
    DATABASES: Database systems, query languages
    CLOUD PLATFORMS: AWS, Azure, GCP, other cloud services
    SOFT SKILLS: Leadership, communication, teamwork, problem-solving
    DOMAIN EXPERTISE: Industry-specific knowledge, business domains
    
    RULES:
    1. Extract both explicitly mentioned and reasonably implied skills
    2. Standardize skill names (e.g., "JS" → "JavaScript", "ML" → "Machine Learning")
    3. Don't repeat skills across categories
    4. Include proficiency levels only if explicitly mentioned
    5. Focus on skills relevant to professional work
    6. Group related skills appropriately
    """


class JobMatchingPrompts:
    """Prompts for job matching and analysis."""
    
    ANALYZE_JOB_MATCH = """
    You are an expert career consultant. Analyze how well this resume matches the job requirements.
    
    Resume Information:
    {resume_data}
    
    Job Information:
    {job_description}
    
    Provide a comprehensive match analysis covering:
    
    OVERALL MATCH:
    - Calculate overall match percentage (0-100)
    - Consider skills, experience, education, and other factors
    
    SKILL ANALYSIS:
    - List matched skills between resume and job requirements
    - Identify missing critical skills
    - Note nice-to-have skills that are missing
    
    EXPERIENCE ANALYSIS:
    - Compare years of experience required vs candidate's experience
    - Assess relevance of candidate's experience to the role
    - Evaluate industry/domain match
    
    EDUCATION ANALYSIS:
    - Compare education requirements with candidate's background
    - Assess field of study relevance
    
    STRENGTHS & GAPS:
    - Highlight candidate's strongest selling points for this role
    - Identify key gaps or weaknesses
    - Provide specific recommendations for improvement
    
    MATCH SCORING:
    Calculate match percentage based on:
    - Required skills match (40% weight)
    - Experience level and relevance (30% weight)
    - Education requirements (15% weight)
    - Industry/domain alignment (15% weight)
    """


class ATSPrompts:
    """Prompts for ATS scoring and optimization."""
    
    CALCULATE_ATS_SCORE = """
    You are an ATS (Applicant Tracking System) expert. Analyze this resume for ATS compatibility and scoring.
    
    Resume Content:
    {resume_text}
    
    Job Description (for keyword analysis):
    {job_description}
    
    Evaluate and score the resume on these ATS factors:
    
    1. KEYWORD OPTIMIZATION (30% weight):
       - Presence of job-relevant keywords from the job description
       - Keyword density and natural usage (not keyword stuffing)
       - Industry-specific terminology and acronyms
       - Action verbs and skill-related terms
    
    2. FORMAT AND STRUCTURE (25% weight):
       - Clear, standard section headers (Experience, Education, Skills, etc.)
       - Consistent formatting throughout the document
       - Proper use of bullet points and spacing
       - Standard resume sections present and well-organized
    
    3. CONTENT QUALITY (20% weight):
       - Quantified achievements with specific numbers/percentages
       - Strong action verbs to start bullet points
       - Relevant experience clearly highlighted
       - Skills section clearly listed and categorized
    
    4. TECHNICAL COMPATIBILITY (15% weight):
       - Text readability (assuming parsed from PDF/DOCX)
       - Use of standard fonts and formatting
       - Minimal use of complex graphics, tables, or unusual layouts
       - Proper text encoding without special characters
    
    5. LENGTH AND CONCISENESS (10% weight):
       - Appropriate length for experience level (1-2 pages typically)
       - Concise, impactful descriptions
       - No unnecessary or irrelevant information
       - Well-organized content hierarchy
    
    ANALYSIS REQUIREMENTS:
    - Provide detailed scoring for each category
    - Calculate overall ATS score (0-100)
    - Identify missing keywords from the job description
    - List specific suggestions for improvement
    - Assign ATS-friendly rating: Excellent (90-100), Good (75-89), Fair (60-74), Poor (0-59)
    
    Focus on actionable feedback that will improve ATS performance.
    """


class ImprovementPrompts:
    """Prompts for resume improvement suggestions."""
    
    GENERATE_IMPROVEMENTS = """
    You are a professional resume coach with expertise in modern hiring practices and ATS optimization.
    
    Current Resume Analysis:
    {resume_analysis}
    
    Target Job Requirements:
    {job_requirements}
    
    ATS Score Analysis:
    {ats_analysis}
    
    Provide comprehensive, actionable improvement recommendations organized by priority:
    
    CRITICAL IMPROVEMENTS (must address):
    - Missing required skills or qualifications
    - Major format issues affecting ATS parsing
    - Significant content gaps or weaknesses
    
    HIGH PRIORITY IMPROVEMENTS (should address):
    - Content enhancement opportunities
    - Keyword optimization for better ATS scoring
    - Quantification of achievements
    - Skills presentation improvements
    
    MEDIUM PRIORITY IMPROVEMENTS (nice to have):
    - Format and visual appeal enhancements
    - Additional keyword incorporation
    - Section organization improvements
    - Content flow optimization
    
    LOW PRIORITY IMPROVEMENTS (polish):
    - Minor formatting consistency
    - Optional section additions
    - Style and presentation refinements
    
    For each improvement, provide:
    - Clear description of the issue
    - Specific recommendation with examples
    - Expected impact on job matching/ATS score
    - Implementation difficulty and time estimate
    - Resources needed (if any)
    
    SKILL DEVELOPMENT PLAN:
    - Immediate actions (can implement today)
    - Short-term goals (1-3 months)
    - Long-term development areas (3-12 months)
    
    ESTIMATED IMPACT:
    - Potential ATS score improvement
    - Expected increase in job match percentage
    - Timeline for implementing changes
    
    Focus on specific, actionable recommendations that will have measurable impact on job search success.
    """


class WorkflowPrompts:
    """Prompts for workflow coordination and decision making."""
    
    WORKFLOW_DECISION = """
    Based on the current analysis state, determine the next best action in the resume analysis workflow.
    
    Current State:
    {current_state}
    
    Available Actions:
    - parse_resume: Extract structured data from resume text
    - find_jobs: Search for relevant job opportunities
    - calculate_ats_score: Compute ATS compatibility score against job description
    - generate_improvements: Create prioritized improvement recommendations
    - complete_analysis: Finish the workflow and compile final report
    
    Consider:
    - What information is already available
    - What information is still needed
    - The logical sequence of operations
    - Dependencies between different analysis steps
    
    Provide the next recommended action with clear reasoning.
    """