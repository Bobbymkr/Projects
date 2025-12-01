# Contributing Guide
## Adaptive Traffic Signal Control System

Thank you for your interest in contributing! This guide will help you get started.

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes
4. **Make your changes** following our guidelines
5. **Submit a pull request**

---

## Development Setup

See [Getting Started Guide](GETTING_STARTED.md) for detailed setup instructions.

Quick setup:
```bash
# Setup environment
python -m src.devtools.cli setup --install-deps

# Verify setup
python -m src.devtools.cli test
```

---

## Code Style

### Python

We follow PEP 8 with some modifications:
- Maximum line length: 100 characters
- Use type hints for all functions
- Docstrings for all public functions/classes

### Formatting

```bash
# Auto-format code
python -m src.devtools.cli format

# Or manually with black
black src/
```

### Linting

```bash
# Check code quality
python -m src.devtools.cli lint

# Auto-fix issues
python -m src.devtools.cli lint --fix
```

---

## Testing

### Running Tests

```bash
# All tests
python -m src.devtools.cli test

# With coverage
python -m src.devtools.cli test --coverage

# Specific tests
pytest tests/specific_test.py
```

### Writing Tests

- Write tests for all new features
- Aim for 90%+ code coverage
- Use descriptive test names
- Follow Arrange-Act-Assert pattern

Example:
```python
def test_dqn_agent_selection():
    """Test DQN agent action selection."""
    # Arrange
    agent = DQNAgent(state_dim=10, action_dim=4)
    state = np.random.rand(10)
    
    # Act
    action = agent.select_action(state)
    
    # Assert
    assert 0 <= action < 4
```

---

## Pull Request Process

### Before Submitting

1. ✅ All tests pass
2. ✅ Code is formatted and linted
3. ✅ Documentation is updated
4. ✅ No merge conflicts

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
- [ ] No merge conflicts
```

---

## Commit Messages

Follow conventional commits:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

Example:
```
feat(research): add federated learning coordinator

Implement federated averaging for distributed learning
across multiple intersections with privacy guarantees.

Closes #123
```

---

## Documentation

### Updating Docs

- Keep documentation in sync with code
- Update docstrings when changing functions
- Add examples for new features
- Update guides when workflows change

### Documentation Structure

```
docs/
├── developer/          # Developer guides
├── api/               # API documentation
├── user/              # User guides
└── research/          # Research documentation
```

---

## Code Review

### Reviewer Guidelines

- Be constructive and respectful
- Focus on code quality and maintainability
- Check tests and documentation
- Approve or request changes clearly

### Author Guidelines

- Respond to feedback promptly
- Make requested changes or explain why not
- Keep discussions focused
- Update PR as needed

---

## Questions?

- 📧 Email: [support email]
- 💬 Slack: [slack channel]
- 🐛 Issues: [GitHub issues]

---

**Thank you for contributing!** 🎉

