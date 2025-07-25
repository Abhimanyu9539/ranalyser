# Tests

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
