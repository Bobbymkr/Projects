# Frequently Asked Questions (FAQ)
## Adaptive Traffic Signal Control System

Common questions and answers for developers and users.

---

## General Questions

### What is the Adaptive Traffic Signal Control System?

The Adaptive Traffic Signal Control System is an AI-powered solution that optimizes traffic signal timing using Deep Reinforcement Learning, Computer Vision, and Multi-Agent coordination to reduce wait times and improve traffic flow.

### What programming languages are used?

- **Backend**: Python 3.10+
- **Frontend**: TypeScript/JavaScript (React)
- **ML/AI**: Python (PyTorch)

### What are the system requirements?

**Minimum**:
- Python 3.10+
- 8GB RAM
- Windows 10/11, Linux, or macOS

**Recommended**:
- Python 3.11+
- 16GB RAM
- GPU for faster ML training
- Docker for containerized deployment

---

## Setup & Installation

### How do I install the project?

See the [Getting Started Guide](GETTING_STARTED.md) for detailed instructions.

Quick install:
```bash
git clone <repository-url>
cd adaptive_traffic
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Why am I getting import errors?

Make sure you:
1. Created and activated the virtual environment
2. Installed all requirements: `pip install -r requirements.txt`
3. Are running from the project root directory
4. Have added the project to PYTHONPATH if needed

### How do I start the API server?

```bash
# Using the provided script
python scripts/start_api.py

# Or directly with uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### How do I start the dashboard?

```bash
cd dashboard
npm install
npm start
```

---

## Development

### How do I add a new algorithm?

See the [Algorithm Implementation Guide](../developer/TUTORIALS.md#algorithm-implementation-guide) for step-by-step instructions.

Basic steps:
1. Create algorithm class in `src/control/` or `src/rl/agents/`
2. Implement required interface methods
3. Register in service layer
4. Add tests
5. Update documentation

### How do I run tests?

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/unit/control/test_fuzzy_control.py
```

### How do I check code quality?

```bash
# Using the CLI
python -m src.devtools.cli lint
python -m src.devtools.cli format

# Or directly
python scripts/quality_check.py
```

### What coding style should I follow?

Follow **PEP 8** with modifications:
- Maximum line length: 100 characters
- Type hints required
- Google-style docstrings
- See [Best Practices](BEST_PRACTICES.md) for details

---

## API Usage

### How do I authenticate with the API?

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    data={"username": "admin", "password": "secret"}
)
token = response.json()["access_token"]
```

### How do I make a traffic decision via API?

```python
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/api/v1/traffic/decision",
    headers=headers,
    json={
        "intersection_id": "intersection_1",
        "current_state": {
            "queue_lengths": [5, 3, 8, 2],
            "wait_times": [12.5, 8.3, 15.2, 6.1]
        },
        "algorithm": "dqn"
    }
)
```

### What algorithms are available?

- `dqn`: Deep Q-Network (default)
- `fuzzy`: Fuzzy Logic Control
- `webster`: Webster's Method
- `marl`: Multi-Agent Reinforcement Learning
- `hierarchical`: Hierarchical RL

---

## Training & Models

### How do I train a new DQN model?

```bash
# Quick training (5 episodes)
python src/rl/train_dqn.py --episodes 5

# Production training (6000 episodes)
python src/rl/train_dqn.py --episodes 6000 --out runs/production
```

### Where are trained models saved?

Models are saved to the `runs/` directory:
- `runs/dqn/best_model.pt`: Best model checkpoint
- `runs/dqn/checkpoints/`: Training checkpoints
- `runs/dqn/tensorboard/`: TensorBoard logs

### How do I use a trained model?

Models are automatically loaded by the service layer. Set the model path in configuration:

```python
config = {
    "model_path": "runs/production/best_model.pt"
}
```

---

## Troubleshooting

### API server won't start

1. Check if port 8000 is available
2. Verify all dependencies are installed
3. Check logs in `logs/` directory
4. Ensure database/Redis are running if required

### Training is slow

1. Use GPU if available (CUDA)
2. Reduce batch size
3. Use smaller network architecture
4. Reduce number of episodes for testing

### Import errors in tests

1. Ensure you're in the project root
2. Install test dependencies: `pip install -r requirements-test.txt`
3. Run: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

### Dashboard won't connect to API

1. Verify API is running on correct port
2. Check CORS settings in API configuration
3. Verify API URL in dashboard config
4. Check browser console for errors

---

## Performance

### What are typical performance metrics?

- **Decision Latency**: < 50ms
- **API Response Time**: < 100ms
- **Wait Time Reduction**: 30-40% vs baseline
- **Throughput Increase**: 25-35%

### How do I optimize performance?

1. Use GPU for ML inference
2. Enable Redis caching
3. Use batch processing for multiple intersections
4. Optimize neural network architecture
5. Use connection pooling

---

## Contributing

### How do I contribute?

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Write/update tests
5. Update documentation
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### What should I include in a PR?

- Clear description of changes
- Tests for new features
- Updated documentation
- Passing CI checks
- No merge conflicts

### How do I report a bug?

Use the [Bug Report template](../../.github/ISSUE_TEMPLATE/bug_report.md) and include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment information
- Error messages/logs

---

## Deployment

### How do I deploy to production?

See the [Deployment Documentation](../deployment/README.md) for detailed instructions.

Basic steps:
1. Configure environment variables
2. Build Docker containers
3. Deploy with Kubernetes or Docker Compose
4. Configure monitoring and logging

### What environment variables are required?

- `SECRET_KEY`: Secret key for authentication
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string (optional)
- `API_HOST`, `API_PORT`: API server configuration

---

## Research & Academic Use

### Can I use this for research?

Yes! The project is open-source and suitable for research. See [LICENSE](../../LICENSE) for details.

### How do I cite this project?

```bibtex
@software{adaptive_traffic_control,
  title={Adaptive Traffic Signal Control System},
  author={...},
  year={2024},
  url={https://github.com/...}
}
```

### Are there published papers?

Check the [Research Documentation](../research/README.md) for published papers and research materials.

---

## Support

### Where can I get help?

- **Documentation**: Check the [docs/](../) directory
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: [support email if available]

### How do I report a security vulnerability?

Please see [SECURITY.md](../../SECURITY.md) for security reporting procedures.

---

**Last Updated**: November 30, 2024

