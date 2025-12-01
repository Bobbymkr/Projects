# Getting Started Guide
## Adaptive Traffic Signal Control System

Welcome to the Adaptive Traffic Signal Control System! This guide will help you get started with development.

---

## Prerequisites

### Required
- **Python 3.10+** (3.11+ recommended)
- **Node.js 18+** (for dashboard development)
- **Git** for version control

### Recommended
- **Docker & Docker Compose** (for containerized development)
- **PostgreSQL** (for production database)
- **Redis** (for caching and rate limiting)

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd adaptive_traffic
```

### 2. Setup Python Environment

#### Option A: Using Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt
```

#### Option B: Using Developer CLI

```bash
# Setup environment automatically
python -m src.devtools.cli setup --install-deps
```

### 3. Setup Dashboard (Optional)

```bash
cd dashboard
npm install
npm start
```

### 4. Run Tests

```bash
# Using CLI
python -m src.devtools.cli test

# Or directly with pytest
pytest tests/
```

### 5. Start Development Servers

```bash
# Start API server
python -m src.devtools.cli server start --service api

# Or directly
python scripts/start_api.py

# Start dashboard (in separate terminal)
cd dashboard && npm start
```

---

## Project Structure

```
adaptive_traffic/
├── src/                    # Python source code
│   ├── api/               # FastAPI application
│   ├── rl/                # Reinforcement learning
│   ├── research/          # Research platform (Phase 5)
│   ├── devtools/          # Developer tools (Phase 6)
│   └── ...
├── dashboard/             # React dashboard
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── requirements*.txt      # Python dependencies
```

---

## Development Workflow

### 1. Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   python -m src.devtools.cli test
   ```

4. Run linting:
   ```bash
   python -m src.devtools.cli lint
   ```

5. Format code:
   ```bash
   python -m src.devtools.cli format
   ```

### 2. Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test
pytest tests/test_specific.py

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/
```

### 3. Code Quality

```bash
# Lint code
python -m src.devtools.cli lint

# Auto-fix linting issues
python -m src.devtools.cli lint --fix

# Format code
python -m src.devtools.cli format
```

---

## Common Tasks

### Running the API

```bash
# Development mode
python scripts/start_api.py

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Database Operations

```bash
# Run migrations
python -m src.devtools.cli db migrate

# Reset database
python -m src.devtools.cli db reset

# Seed database
python -m src.devtools.cli db seed
```

### Generating Code

```bash
# Generate new component
python -m src.devtools.cli generate component MyComponent

# Generate test file
python -m src.devtools.cli generate test test_my_component

# Generate API endpoint
python -m src.devtools.cli generate api my_endpoint
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Database
DATABASE_URL=postgresql://user:password@localhost/adaptive_traffic

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
```

---

## Troubleshooting

### Common Issues

#### Virtual Environment Issues
```bash
# If activation fails, try:
python -m venv .venv --clear
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

#### Dependency Issues
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Then reinstall dependencies
pip install -r requirements.txt
```

#### Import Errors
```bash
# Ensure you're in the project root
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Next Steps

- 📖 Read the [Architecture Guide](ARCHITECTURE.md)
- 📚 Check out [API Documentation](../api/README.md)
- 🧪 Review [Testing Guide](TESTING.md)
- 🤝 Read [Contribution Guidelines](CONTRIBUTING.md)

---

## Getting Help

- 📧 Email: [support email]
- 💬 Slack: [slack channel]
- 📖 Documentation: [docs link]
- 🐛 Issues: [GitHub issues]

---

**Happy Coding!** 🚀

