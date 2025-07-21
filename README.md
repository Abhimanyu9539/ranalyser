# Resume Analyzer

A comprehensive resume analysis and job matching system using LangGraph, LangChain, and OpenAI APIs.

## Features

- Resume parsing from PDF/DOCX files
- Intelligent job matching
- ATS score calculation
- Personalized improvement recommendations

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
python main.py
```

## API Usage

Start the web server:
```bash
cd api
python app.py
```

Navigate to http://localhost:8000

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black .
isort .
```

## Documentation

See `docs/` directory for detailed documentation.
