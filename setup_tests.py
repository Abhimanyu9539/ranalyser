#!/usr/bin/env python3
"""
Script to set up the test structure for the Resume Analyzer project.
Run this to create all necessary test directories and files.
"""
import os
from pathlib import Path


def create_directory_structure():
    """Create the test directory structure."""
    base_path = Path("tests")
    
    directories = [
        "tests",
        "tests/data",
        "tests/data/sample_resumes", 
        "tests/data/sample_jobs",
        "tests/unit",
        "tests/integration",
        "tests/mocks",
        "tests/utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_init_files():
    """Create __init__.py files to make directories Python packages."""
    init_files = [
        "tests/__init__.py",
        "tests/data/__init__.py", 
        "tests/unit/__init__.py",
        "tests/integration/__init__.py",
        "tests/mocks/__init__.py",
        "tests/utils/__init__.py"
    ]
    
    init_content = '"""\nTest package.\n"""\n'
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write(init_content)
        print(f"✅ Created: {init_file}")


def create_pytest_config():
    """Create pytest.ini configuration file."""
    pytest_config = """[tool:pytest]
# pytest.ini - Pytest configuration

[pytest]
# Test discovery
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Markers for categorizing tests
markers =
    unit: Unit tests (no external dependencies)
    integration: Integration tests (require API keys)
    slow: Slow tests (may take longer to run)
    smoke: Smoke tests (basic functionality)

# Output options
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings

# Async test support
asyncio_mode = auto

# Test timeout (in seconds)
timeout = 300

# Logging
log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(name)s: %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
"""
    
    with open("pytest.ini", 'w') as f:
        f.write(pytest_config)
    print("✅ Created: pytest.ini")


def create_sample_files():
    """Create sample test data files."""
    
    # Sample resume
    sample_resume = """Senior Software Engineer
Email: john.doe@email.com | Phone: (555) 123-4567

EXPERIENCE:
Senior Software Engineer at TechCorp (2020-Present)
• Led development of microservices using Python and Django
• Managed team of 8 developers using Agile methodologies
• 5+ years experience with React and JavaScript

SKILLS:
Programming: Python, JavaScript, TypeScript
Frameworks: Django, React, Node.js
Databases: PostgreSQL, MongoDB, Redis
Cloud: AWS, Docker, Kubernetes
"""
    
    with open("tests/data/sample_resumes/senior_developer.txt", 'w') as f:
        f.write(sample_resume)
    print("✅ Created: tests/data/sample_resumes/senior_developer.txt")
    
    # Sample job description
    sample_job = """Senior Full Stack Developer

REQUIRED QUALIFICATIONS:
• 5+ years of Python development experience
• Strong experience with Django or Flask
• Frontend development with React
• Database design with PostgreSQL
• Experience with Docker and CI/CD

PREFERRED QUALIFICATIONS:
• AWS cloud platform experience
• Team leadership experience
• TypeScript knowledge
"""
    
    with open("tests/data/sample_jobs/fullstack_senior.txt", 'w') as f:
        f.write(sample_job)
    print("✅ Created: tests/data/sample_jobs/fullstack_senior.txt")


def create_readme():
    """Create README for the tests directory."""
    readme_content = """# Tests

This directory contains comprehensive tests for the Resume Analyzer skills extraction system.

## Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_skills_extractor.py # Main test file (moved from artifacts)
├── test_runner.py           # Manual test runner
├── data/                    # Test data and samples
│   ├── sample_resumes/      # Sample resume files
│   ├── sample_jobs/         # Sample job descriptions
│   └── skill_test_cases.json # Test cases data
├── unit/                    # Unit tests (no API calls)
├── integration/            # Integration tests (require API)
├── mocks/                  # Mock data and responses
└── utils/                  # Test utilities and helpers

## Running Tests

### Unit Tests (No API Key Required)
```bash
pytest -m "not integration"
```

### Integration Tests (Requires OPENAI_API_KEY)
```bash
export OPENAI_API_KEY="your-key-here"
pytest -m integration
```

### All Tests
```bash
pytest
```

### Manual Testing
```bash
python tests/test_runner.py quick      # Quick functionality test
python tests/test_runner.py manual     # Comprehensive testing
python tests/test_runner.py all        # All manual tests
```

## Test Categories

- **Unit Tests**: Fast tests that don't require external APIs
- **Integration Tests**: Tests with real LLM calls (require API key)
- **Performance Tests**: Measure extraction speed and quality
- **Context Tests**: Verify context-specific behavior

## Adding New Tests

1. **Unit tests**: Add to `tests/unit/`
2. **Integration tests**: Add to `tests/integration/`
3. **Test data**: Add to `tests/data/`
4. **Mock responses**: Add to `tests/mocks/`

## Prerequisites

1. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_MODEL="gpt-4"  # optional
   ```

3. Run tests from project root:
   ```bash
   pytest tests/
   ```
"""
    
    with open("tests/README.md", 'w') as f:
        f.write(readme_content)
    print("✅ Created: tests/README.md")


def create_gitkeep_files():
    """Create .gitkeep files for empty directories."""
    gitkeep_dirs = [
        "tests/data/sample_resumes",
        "tests/data/sample_jobs"
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_path = Path(directory) / ".gitkeep"
        gitkeep_path.touch()
        print(f"✅ Created: {gitkeep_path}")


def main():
    """Main setup function."""
    print("🔧 Setting up Resume Analyzer test structure...")
    print()
    
    # Create directory structure
    create_directory_structure()
    print()
    
    # Create __init__.py files
    create_init_files()
    print()
    
    # Create configuration files
    create_pytest_config()
    print()
    
    # Create sample data files
    create_sample_files()
    print()
    
    # Create documentation
    create_readme()
    print()
    
    # Create .gitkeep files
    create_gitkeep_files()
    print()
    
    print("✅ Test structure setup complete!")
    print()
    print("📋 Next steps:")
    print("1. Copy the test files from the artifacts into their respective directories")
    print("2. Install test dependencies: pip install pytest pytest-asyncio")
    print("3. Set OPENAI_API_KEY environment variable for integration tests")
    print("4. Run tests: pytest tests/")
    print()
    print("📖 See tests/README.md for detailed instructions")


if __name__ == "__main__":
    main()