# tests/mocks/mock_llm_responses.py
"""
Mock LLM responses for testing without API calls.
"""

MOCK_RESUME_RESPONSE = {
    "skills": [
        {
            "name": "Python",
            "category": "programming_languages",
            "confidence": "high",
            "evidence": ["5+ years experience with Python", "Led development using Python"],
            "years_experience": 5.0,
            "proficiency_level": "Expert",
            "context": "Work experience section"
        },
        {
            "name": "Django",
            "category": "frameworks_libraries",
            "confidence": "high",
            "evidence": ["microservices architecture using Python and Django"],
            "years_experience": None,
            "proficiency_level": "Advanced",
            "context": "Technical implementation"
        },
        {
            "name": "React",
            "category": "frameworks_libraries",
            "confidence": "high",
            "evidence": ["5+ years experience with React and JavaScript"],
            "years_experience": 5.0,
            "proficiency_level": "Expert",
            "context": "Frontend development"
        },
        {
            "name": "Leadership",
            "category": "soft_skills",
            "confidence": "high",
            "evidence": ["Managed team of 8 developers", "Led development"],
            "years_experience": 3.0,
            "proficiency_level": None,
            "context": "Management experience"
        },
        {
            "name": "PostgreSQL",
            "category": "databases",
            "confidence": "medium",
            "evidence": ["REST APIs using Django REST Framework and PostgreSQL"],
            "years_experience": None,
            "proficiency_level": "Intermediate",
            "context": "Database usage"
        },
        {
            "name": "Docker",
            "category": "tools_software",
            "confidence": "medium",
            "evidence": ["CI/CD pipelines with Jenkins and Docker"],
            "years_experience": None,
            "proficiency_level": "Intermediate",
            "context": "DevOps practices"
        },
        {
            "name": "Agile",
            "category": "technical_skills",
            "confidence": "high",
            "evidence": ["team using Agile methodologies"],
            "years_experience": None,
            "proficiency_level": None,
            "context": "Development methodology"
        }
    ]
}

MOCK_JOB_RESPONSE = {
    "skills": [
        {
            "name": "Python",
            "category": "programming_languages",
            "confidence": "high",
            "evidence": ["Expert-level Python programming"],
            "years_experience": 5.0,
            "proficiency_level": "Expert",
            "context": "Required qualification"
        },
        {
            "name": "Django",
            "category": "frameworks_libraries",
            "confidence": "high",
            "evidence": ["Python programming with Django or Flask"],
            "years_experience": None,
            "proficiency_level": "Expert",
            "context": "Required framework"
        },
        {
            "name": "Flask",
            "category": "frameworks_libraries", 
            "confidence": "high",
            "evidence": ["Python programming with Django or Flask"],
            "years_experience": None,
            "proficiency_level": "Expert",
            "context": "Alternative framework"
        },
        {
            "name": "React",
            "category": "frameworks_libraries",
            "confidence": "high",
            "evidence": ["Strong frontend skills with React"],
            "years_experience": None,
            "proficiency_level": "Strong",
            "context": "Required frontend skill"
        },
        {
            "name": "PostgreSQL",
            "category": "databases",
            "confidence": "high",
            "evidence": ["Experience with PostgreSQL and database design"],
            "years_experience": None,
            "proficiency_level": "Experienced",
            "context": "Required database"
        },
        {
            "name": "Docker",
            "category": "tools_software",
            "confidence": "high",
            "evidence": ["Proficiency with Git, Docker, and CI/CD practices"],
            "years_experience": None,
            "proficiency_level": "Proficient",
            "context": "Required tool"
        },
        {
            "name": "AWS",
            "category": "cloud_platforms",
            "confidence": "medium",
            "evidence": ["AWS cloud platform experience"],
            "years_experience": None,
            "proficiency_level": "Experienced",
            "context": "Preferred qualification"
        },
        {
            "name": "TypeScript",
            "category": "programming_languages",
            "confidence": "medium", 
            "evidence": ["TypeScript and modern frontend tooling"],
            "years_experience": None,
            "proficiency_level": "Familiar",
            "context": "Preferred skill"
        },
        {
            "name": "Team Leadership",
            "category": "soft_skills",
            "confidence": "medium",
            "evidence": ["Previous team leadership experience"],
            "years_experience": None,
            "proficiency_level": None,
            "context": "Preferred experience"
        }
    ]
}


