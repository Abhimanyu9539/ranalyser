#!/usr/bin/env python3
"""
Test runner script for skills extractor with different test modes.
"""
import os
import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.skills_extractor import LLMSkillExtractor, extract_skills_from_resume


class SkillExtractorTester:
    """Manual testing utility for skills extractor."""
    
    def __init__(self):
        self.extractor = LLMSkillExtractor()
        self.test_results = []
    
    async def run_manual_tests(self):
        """Run manual tests with real examples."""
        print("🧪 Running Manual Skills Extraction Tests\n")
        
        test_cases = [
            {
                "name": "Senior Software Engineer Resume",
                "text": """
                Senior Software Engineer with 8 years of experience in full-stack development.
                
                TECHNICAL SKILLS:
                • Expert in Python (Django, Flask, FastAPI) - 8 years
                • Advanced JavaScript/TypeScript (React, Node.js) - 6 years  
                • Proficient in Go and Java - 3 years
                • Database: PostgreSQL, MongoDB, Redis
                • Cloud: AWS (EC2, S3, Lambda), Docker, Kubernetes
                • DevOps: Jenkins, GitLab CI, Terraform
                
                EXPERIENCE:
                Lead Software Engineer at TechCorp (2020-Present)
                • Led team of 12 developers across 3 product lines
                • Architected microservices platform handling 1M+ daily users
                • Implemented CI/CD pipelines reducing deployment time by 80%
                • Mentored 5 junior developers and conducted technical interviews
                
                Software Engineer at StartupXYZ (2018-2020)
                • Built real-time chat application using WebSocket and Redis
                • Developed RESTful APIs serving 100K+ requests per day
                • Optimized database queries improving performance by 40%
                """,
                "context": "resume",
                "expected_categories": ["programming_languages", "frameworks_libraries", "tools_software", "soft_skills"]
            },
            {
                "name": "Job Description - Senior Full Stack Developer",
                "text": """
                Senior Full Stack Developer - Remote
                
                REQUIRED QUALIFICATIONS:
                • 5+ years of professional software development experience
                • Expert-level Python programming with Django or Flask
                • Strong frontend skills with React and modern JavaScript
                • Experience with PostgreSQL and database design
                • Proficiency with Git, Docker, and CI/CD practices
                • Experience leading technical projects and mentoring team members
                
                PREFERRED QUALIFICATIONS:
                • AWS cloud platform experience (EC2, S3, RDS, Lambda)
                • TypeScript and modern frontend tooling (Webpack, Babel)
                • Experience with microservices architecture
                • Knowledge of DevOps practices and infrastructure as code
                • Previous startup experience and agile development methodologies
                
                RESPONSIBILITIES:
                • Design and implement scalable web applications
                • Lead architectural decisions and code reviews
                • Collaborate with product and design teams
                • Mentor junior developers and participate in hiring
                """,
                "context": "job_description",
                "expected_categories": ["programming_languages", "frameworks_libraries", "cloud_platforms", "soft_skills"]
            },
            {
                "name": "Project Description - E-commerce Platform",
                "text": """
                E-commerce Platform Rebuild Project
                
                OVERVIEW:
                Complete rebuild of legacy e-commerce platform to modern microservices architecture.
                
                TECHNOLOGY STACK:
                Backend: Python 3.9, Django 4.0, Django REST Framework
                Frontend: React 18, TypeScript, Redux Toolkit, Material-UI
                Database: PostgreSQL 14, Redis for caching
                Search: Elasticsearch 8.0
                Message Queue: RabbitMQ with Celery
                API Gateway: Kong
                Infrastructure: AWS (ECS, RDS, ElastiCache, S3)
                CI/CD: GitHub Actions, Docker, Terraform
                Monitoring: Prometheus, Grafana, Sentry
                
                ARCHITECTURE:
                • Microservices with API Gateway pattern
                • Event-driven architecture using RabbitMQ
                • Containerized deployment with Docker
                • Infrastructure as Code with Terraform
                • Automated testing and deployment pipelines
                
                DEVELOPMENT PRACTICES:
                • Test-Driven Development (TDD)
                • Code reviews and pair programming
                • Agile/Scrum methodology
                • Continuous integration and deployment
                """,
                "context": "project_description",
                "expected_categories": ["programming_languages", "frameworks_libraries", "databases", "cloud_platforms", "tools_software"]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"📋 Test {i}: {test_case['name']}")
            print("-" * 60)
            
            try:
                result = await self.extractor.extract_skills_from_text(
                    test_case["text"], 
                    test_case["context"]
                )
                
                await self._analyze_test_result(test_case, result)
                print()
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                print()
        
        self._print_summary()
    
    async def _analyze_test_result(self, test_case, result):
        """Analyze and print test results."""
        print(f"⏱️  Processing time: {result.extraction_metadata.get('processing_time', 0):.2f}s")
        print(f"📊 Total skills found: {len(result.skills)}")
        print(f"🎯 High confidence: {len(result.get_high_confidence_skills())}")
        
        # Group by category
        print("\n📂 Skills by Category:")
        categories_found = {}
        for skill in result.skills:
            if skill.category not in categories_found:
                categories_found[skill.category] = []
            categories_found[skill.category].append(skill)
        
        for category, skills in categories_found.items():
            print(f"   {category.value}: {len(skills)} skills")
            for skill in skills[:3]:  # Show first 3
                confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[skill.confidence.value]
                print(f"     {confidence_icon} {skill.name}")
            if len(skills) > 3:
                print(f"     ... and {len(skills) - 3} more")
        
        # Show skills with experience
        experienced_skills = result.get_skills_with_experience()
        if experienced_skills:
            print(f"\n⏳ Skills with Experience ({len(experienced_skills)}):")
            for skill in experienced_skills[:5]:
                print(f"   • {skill.name}: {skill.years_experience} years")
        
        # Check coverage of expected categories
        found_category_names = {skill.category.value for skill in result.skills}
        expected_categories = set(test_case.get('expected_categories', []))
        category_coverage = len(found_category_names.intersection(expected_categories)) / len(expected_categories) if expected_categories else 1
        
        print(f"\n📈 Category Coverage: {category_coverage:.1%}")
        
        # Store result for summary
        self.test_results.append({
            "name": test_case["name"],
            "context": test_case["context"],
            "total_skills": len(result.skills),
            "high_confidence": len(result.get_high_confidence_skills()),
            "processing_time": result.extraction_metadata.get('processing_time', 0),
            "category_coverage": category_coverage
        })
    
    def _print_summary(self):
        """Print overall test summary."""
        print("=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        if not self.test_results:
            print("No test results to summarize.")
            return
        
        total_tests = len(self.test_results)
        avg_skills = sum(r["total_skills"] for r in self.test_results) / total_tests
        avg_confidence = sum(r["high_confidence"] for r in self.test_results) / total_tests
        avg_time = sum(r["processing_time"] for r in self.test_results) / total_tests
        avg_coverage = sum(r["category_coverage"] for r in self.test_results) / total_tests
        
        print(f"Tests run: {total_tests}")
        print(f"Average skills extracted: {avg_skills:.1f}")
        print(f"Average high-confidence skills: {avg_confidence:.1f}")
        print(f"Average processing time: {avg_time:.2f}s")
        print(f"Average category coverage: {avg_coverage:.1%}")
        
        print(f"\n📊 Results by Context:")
        contexts = {}
        for result in self.test_results:
            context = result["context"]
            if context not in contexts:
                contexts[context] = []
            contexts[context].append(result)
        
        for context, results in contexts.items():
            avg_skills_context = sum(r["total_skills"] for r in results) / len(results)
            print(f"   {context}: {avg_skills_context:.1f} avg skills")
    
    async def test_comparison_functionality(self):
        """Test skill comparison between resume and job."""
        print("\n🔄 Testing Skill Comparison Functionality")
        print("-" * 60)
        
        resume_text = """
        Full Stack Developer with 4 years experience.
        Skills: Python, Django, React, PostgreSQL, Docker, AWS
        Experience with agile methodologies and team leadership.
        """
        
        job_text = """
        Looking for Full Stack Developer with:
        Required: Python, Django, React, PostgreSQL, 3+ years experience
        Preferred: Docker, AWS, TypeScript, leadership experience
        """
        
        try:
            from src.tools.skills_extractor import compare_resume_job_skills
            comparison = await compare_resume_job_skills(resume_text, job_text)
            
            comp_data = comparison["comparison"]
            print(f"🎯 Match Percentage: {comp_data['match_percentage']}%")
            print(f"✅ Matched Skills: {', '.join(comp_data['matched_skills'])}")
            print(f"❌ Missing Skills: {', '.join(comp_data['missing_skills'])}")
            print(f"➕ Additional Skills: {', '.join(comp_data['additional_skills'])}")
            
        except Exception as e:
            print(f"❌ Comparison test failed: {e}")


async def run_quick_test():
    """Run a quick test to verify the system works."""
    print("🚀 Quick Skills Extraction Test")
    print("-" * 40)
    
    sample_text = """
    Software Engineer with 3 years Python experience. 
    Built web applications using Django and React.
    Experience with PostgreSQL and Docker.
    """
    
    try:
        result = await extract_skills_from_resume(sample_text)
        
        print(f"✅ Success! Found {len(result.skills)} skills:")
        for skill in result.skills[:5]:
            confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[skill.confidence.value]
            print(f"   {confidence_icon} {skill.name} ({skill.category.value})")
        
        if len(result.skills) > 5:
            print(f"   ... and {len(result.skills) - 5} more")
        
        print(f"\n⏱️  Processing time: {result.extraction_metadata.get('processing_time', 0):.2f}s")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure you have:")
        print("   1. Set OPENAI_API_KEY in your environment")
        print("   2. Installed all requirements")
        print("   3. Configured the settings properly")


async def run_context_test():
    """Test context-specific behavior."""
    print("🎭 Context-Specific Behavior Test")
    print("-" * 40)
    
    text = "Led team of 8 developers using Python and Django"
    contexts = ["resume", "job_description", "project_description"]
    
    try:
        extractor = LLMSkillExtractor()
        
        for context in contexts:
            print(f"\n📋 Context: {context}")
            result = await extractor.extract_skills_from_text(text, context)
            
            skills_found = [skill.name for skill in result.skills]
            print(f"   Skills: {', '.join(skills_found)}")
            
            high_conf = [skill.name for skill in result.get_high_confidence_skills()]
            print(f"   High confidence: {', '.join(high_conf)}")
            
    except Exception as e:
        print(f"❌ Context test failed: {e}")


async def run_performance_test():
    """Test performance with different text sizes."""
    print("⚡ Performance Test")
    print("-" * 40)
    
    test_cases = [
        ("Short", "Python developer with Django experience"),
        ("Medium", "Python developer with Django experience. " * 20),
        ("Long", "Python developer with Django experience. " * 100)
    ]
    
    try:
        extractor = LLMSkillExtractor()
        
        for name, text in test_cases:
            result = await extractor.extract_skills_from_text(text, "resume")
            processing_time = result.extraction_metadata.get('processing_time', 0)
            
            print(f"   {name}: {len(result.skills)} skills in {processing_time:.2f}s "
                  f"({len(text)} chars)")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            asyncio.run(run_quick_test())
        elif command == "manual":
            tester = SkillExtractorTester()
            asyncio.run(tester.run_manual_tests())
        elif command == "comparison":
            tester = SkillExtractorTester()
            asyncio.run(tester.test_comparison_functionality())
        elif command == "context":
            asyncio.run(run_context_test())
        elif command == "performance":
            asyncio.run(run_performance_test())
        elif command == "all":
            print("🔬 Running All Manual Tests")
            print("=" * 50)
            
            asyncio.run(run_quick_test())
            print("\n" + "="*60 + "\n")
            
            tester = SkillExtractorTester()
            asyncio.run(tester.run_manual_tests())
            
            print("\n" + "="*60 + "\n")
            asyncio.run(tester.test_comparison_functionality())
            
            print("\n" + "="*60 + "\n")
            asyncio.run(run_context_test())
            
            print("\n" + "="*60 + "\n")
            asyncio.run(run_performance_test())
        else:
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage instructions."""
    print("""
🧪 Skills Extractor Test Runner

Usage: python tests/test_runner.py [command]

Commands:
  quick       - Run a quick test to verify basic functionality
  manual      - Run comprehensive manual tests with detailed analysis
  comparison  - Test skill comparison functionality
  context     - Test context-specific behavior differences
  performance - Test performance with different text sizes
  all         - Run all tests

Examples:
  python tests/test_runner.py quick
  python tests/test_runner.py manual
  python tests/test_runner.py all

Prerequisites:
  1. Set OPENAI_API_KEY environment variable
     export OPENAI_API_KEY="your-api-key-here"
  
  2. Install requirements
     pip install -r requirements.txt
  
  3. Ensure config/settings.py is properly configured

Test Categories:
  🚀 Quick     - Fast verification that system works
  🧪 Manual    - Comprehensive analysis with real examples  
  🔄 Comparison - Resume vs job description matching
  🎭 Context   - Different behavior based on document type
  ⚡ Performance - Speed and efficiency testing

Notes:
  - All tests require an OpenAI API key
  - Tests make real API calls and may incur costs
  - Use 'quick' for development, 'all' for comprehensive testing
""")


if __name__ == "__main__":
    main()