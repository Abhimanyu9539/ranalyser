"""
Prompt templates for the Resume Analyzer application.
"""

class ResumePrompts:
    """Prompts for resume parsing and analysis."""
    
    PARSE_RESUME = """
    You are an expert resume parser. Extract structured information from the following resume text.
    
    Resume Text:
    {resume_text}
    
    Please extract and format the information as JSON with the following structure:
    {{
        "personal_info": {{
            "name": "Full name",
            "email": "email@example.com",
            "phone": "phone number",
            "location": "city, state/country",
            "linkedin": "linkedin profile URL",
            "portfolio": "portfolio/website URL"
        }},
        "summary": "Professional summary or objective",
        "experience": [
            {{
                "title": "Job Title",
                "company": "Company Name",
                "location": "City, State",
                "start_date": "MM/YYYY",
                "end_date": "MM/YYYY or Present",
                "description": "Job description and achievements",
                "key_achievements": ["Achievement 1", "Achievement 2"]
            }}
        ],
        "education": [
            {{
                "degree": "Degree Type",
                "field": "Field of Study",
                "institution": "University/School Name",
                "graduation_date": "MM/YYYY",
                "gpa": "GPA if mentioned",
                "honors": ["Honor 1", "Honor 2"] if mentioned
            }}
        ],
        "skills": {{
            "technical": ["Skill 1", "Skill 2"],
            "programming_languages": ["Language 1", "Language 2"],
            "frameworks_tools": ["Framework 1", "Tool 1"],
            "soft_skills": ["Soft Skill 1", "Soft Skill 2"]
        }},
        "certifications": [
            {{
                "name": "Certification Name",
                "issuer": "Issuing Organization",
                "date": "MM/YYYY",
                "expiry": "MM/YYYY or Never"
            }}
        ],
        "projects": [
            {{
                "name": "Project Name",
                "description": "Project description",
                "technologies": ["Tech 1", "Tech 2"],
                "url": "project URL if available"
            }}
        ]
    }}
    
    Instructions:
    1. Extract only information that is explicitly mentioned in the resume
    2. Use "Not specified" for missing optional fields
    3. Standardize date formats to MM/YYYY
    4. Clean and normalize company names, job titles, and skills. Do not makeup the names.
    5. Group similar skills together logically
    6. Ensure all JSON is valid and properly formatted
    """
    
    EXTRACT_SKILLS = """
    You are a skills extraction expert. Analyze the following resume text and extract all relevant skills.
    
    Resume Text:
    {resume_text}
    
    Categorize skills into these specific categories:
    - Technical Skills (software, platforms, methodologies)
    - Programming Languages
    - Frameworks & Libraries
    - Tools & Software
    - Databases
    - Cloud Platforms
    - Soft Skills (leadership, communication, etc.)
    - Domain Expertise (industry-specific knowledge)
    
    Return as JSON:
    {{
        "technical_skills": [],
        "programming_languages": [],
        "frameworks_libraries": [],
        "tools_software": [],
        "databases": [],
        "cloud_platforms": [],
        "soft_skills": [],
        "domain_expertise": []
    }}
    
    Rules:
    1. Extract both explicitly mentioned and implied skills
    2. Standardize skill names (e.g., "JS" → "JavaScript")
    3. Don't repeat skills across categories
    4. Include proficiency levels if mentioned. Do not include on your own.
    5. Focus on relevant skills
    """

class JobMatchingPrompts:
    """Prompts for job matching and analysis."""
    
    ANALYZE_JOB_MATCH = """
    You are an expert career consultant. Analyze how well this resume matches the job requirements.
    
    Resume Skills and Experience:
    {resume_data}
    
    Job Requirements:
    {job_description}
    
    Provide a detailed match analysis as JSON:
    {{
        "overall_match_percentage": 85,
        "skill_matches": {{
            "matched_skills": ["skill1", "skill2"],
            "missing_critical_skills": ["skill3", "skill4"],
            "nice_to_have_missing": ["skill5"]
        }},
        "experience_match": {{
            "years_required": 5,
            "years_candidate_has": 4,
            "relevant_experience_match": 80,
            "industry_match": true
        }},
        "education_match": {{
            "required_degree": "Bachelor's",
            "candidate_degree": "Bachelor's",
            "field_relevance": 90
        }},
        "strengths": ["strength1", "strength2"],
        "gaps": ["gap1", "gap2"],
        "recommendations": ["recommendation1", "recommendation2"]
    }}
    
    Calculate match percentage based on:
    - Required skills match (40%)
    - Experience level match (30%)
    - Education requirements (15%)
    - Industry/domain match (15%)
    """

class ATSPrompts:
    """Prompts for ATS scoring and optimization."""
    
    CALCULATE_ATS_SCORE = """
    You are an ATS (Applicant Tracking System) expert. Analyze this resume for ATS compatibility.
    
    Resume Content:
    {resume_text}
    
    Job Description (for keyword analysis):
    {job_description}
    
    Evaluate and score the resume on these ATS factors:
    
    1. **Keyword Optimization (30%)**
       - Presence of job-relevant keywords
       - Keyword density and natural usage
       - Industry-specific terminology
    
    2. **Format and Structure (25%)**
       - Clear section headers
       - Consistent formatting
       - Proper use of bullet points
       - Standard resume sections present
    
    3. **Content Quality (20%)**
       - Quantified achievements
       - Action verbs usage
       - Relevant experience highlighted
       - Skills clearly listed
    
    4. **Technical Compatibility (15%)**
       - Text readability (assuming parsed from PDF/DOCX)
       - Standard fonts and formatting
       - No complex graphics or tables
    
    5. **Length and Conciseness (10%)**
       - Appropriate length (1-2 pages)
       - Concise descriptions
       - No unnecessary information
    
    Return detailed scoring as JSON:
    {{
        "overall_ats_score": 78,
        "category_scores": {{
            "keyword_optimization": {{
                "score": 75,
                "max_score": 30,
                "feedback": "Good keyword usage but missing some key terms"
            }},
            "format_structure": {{
                "score": 85,
                "max_score": 25,
                "feedback": "Well-structured with clear sections"
            }},
            "content_quality": {{
                "score": 70,
                "max_score": 20,
                "feedback": "Needs more quantified achievements"
            }},
            "technical_compatibility": {{
                "score": 90,
                "max_score": 15,
                "feedback": "Excellent technical format"
            }},
            "length_conciseness": {{
                "score": 80,
                "max_score": 10,
                "feedback": "Good length and conciseness"
            }}
        }},
        "missing_keywords": ["keyword1", "keyword2"],
        "suggestions": [
            "Add more quantified achievements",
            "Include specific keywords: keyword1, keyword2",
            "Improve action verb usage"
        ],
        "ats_friendly_rating": "Good"
    }}
    
    Rating scale: Excellent (90-100), Good (75-89), Fair (60-74), Poor (0-59)
    """

class ImprovementPrompts:
    """Prompts for resume improvement suggestions."""
    
    GENERATE_IMPROVEMENTS = """
    You are a professional resume coach with expertise in modern hiring practices.
    
    Current Resume Analysis:
    {resume_analysis}
    
    Target Job Requirements:
    {job_requirements}
    
    ATS Score Analysis:
    {ats_analysis}
    
    Provide comprehensive improvement recommendations categorized as follows:
    
    {{
        "critical_improvements": [
            {{
                "category": "Missing Skills",
                "issue": "Lack of required skill X",
                "recommendation": "Add skill X to technical skills section",
                "impact": "High - Required for role",
                "implementation": "Take course Y or highlight existing experience"
            }}
        ],
        "high_priority_improvements": [
            {{
                "category": "Content Enhancement",
                "issue": "Lack of quantified achievements",
                "recommendation": "Add metrics to 3-5 key achievements",
                "impact": "High - Improves credibility",
                "implementation": "Review past work and add specific numbers/percentages"
            }}
        ],
        "medium_priority_improvements": [
            {{
                "category": "Keyword Optimization",
                "issue": "Missing industry keywords",
                "recommendation": "Incorporate keywords: X, Y, Z naturally",
                "impact": "Medium - Improves ATS score",
                "implementation": "Revise job descriptions to include these terms"
            }}
        ],
        "nice_to_have_improvements": [
            {{
                "category": "Format Enhancement",
                "issue": "Could improve visual appeal",
                "recommendation": "Add consistent formatting to sections",
                "impact": "Low - Aesthetic improvement",
                "implementation": "Use consistent bullet points and spacing"
            }}
        ],
        "skill_development_plan": {{
            "immediate_actions": ["Action 1", "Action 2"],
            "short_term_goals": ["Goal 1", "Goal 2"],
            "long_term_development": ["Development area 1", "Development area 2"]
        }},
        "estimated_impact": {{
            "ats_score_improvement": "+15 points",
            "match_percentage_improvement": "+20%",
            "timeline": "2-4 weeks for implementation"
        }}
    }}
    
    Focus on actionable, specific recommendations that will have measurable impact.
    """

class WorkflowPrompts:
    """Prompts for workflow coordination and decision making."""
    
    WORKFLOW_DECISION = """
    Based on the current analysis state, determine the next best action in the resume analysis workflow.
    
    Current State:
    {current_state}
    
    Available Actions:
    - parse_resume: Extract structured data from resume
    - find_jobs: Search for relevant job opportunities
    - calculate_ats_score: Compute ATS compatibility score
    - generate_improvements: Create improvement recommendations
    - complete_analysis: Finish the workflow
    
    Return the next action and reasoning:
    {{
        "next_action": "action_name",
        "reasoning": "Why this action should be taken next",
        "parameters": {{"param1": "value1"}},
        "priority": "high|medium|low"
    }}
    """