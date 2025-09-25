# Contributing to FIAE AI Content Factory

Thank you for your interest in contributing to the FIAE AI Content Factory! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for containerized development)
- Google Cloud credentials (for testing)

### Development Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/your-username/AI-Content-Generation-.git
cd AI-Content-Generation-
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

4. **Set up environment variables**:
```bash
cp env.example .env
# Edit .env with your configuration
```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Use meaningful variable and function names

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

### Documentation
- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update API documentation if endpoints change

## 🔄 Pull Request Process

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**:
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
```bash
pytest tests/
black app/
flake8 app/
mypy app/
```

4. **Commit your changes**:
```bash
git add .
git commit -m "Add: Brief description of changes"
```

5. **Push and create PR**:
```bash
git push origin feature/your-feature-name
```

## 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Logs or error messages

## 💡 Feature Requests

For feature requests, please:
- Check existing issues first
- Provide clear use case and motivation
- Describe the expected behavior
- Consider implementation complexity

## 🏗️ Project Structure

```
FIAE-Agents-with-RAG/
├── app/                    # Main application code
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration management
│   ├── models.py          # Pydantic models
│   └── services/          # Core AI services
├── tests/                 # Test files
├── docs/                  # Documentation
├── n8n-workflows/         # n8n workflow files
├── credentials/           # Credentials (gitignored)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── docker-compose.yml    # Docker services
```

## 🔧 Development Tools

### Code Quality
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

## 📋 Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority: high`: High priority issue
- `priority: low`: Low priority issue

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the golden rule

## 📞 Getting Help

- Check existing issues and discussions
- Join our community discussions
- Contact maintainers for urgent issues

## 🎯 Areas for Contribution

- **AI/ML Improvements**: Enhance RAG processing, add new models
- **API Enhancements**: New endpoints, better error handling
- **Documentation**: Improve guides, add examples
- **Testing**: Increase test coverage, add integration tests
- **Performance**: Optimize processing speed, reduce memory usage
- **Monitoring**: Enhance observability, add metrics
- **UI/UX**: Improve user interfaces and workflows

## 📄 License

This is proprietary software. All rights reserved.

---

Thank you for contributing to FIAE AI Content Factory! 🚀
