# Contributing Guide

Thank you for your interest in contributing to Dream Trainer! This guide will help you get started.

---

## Ways to Contribute

- **Bug reports**: Found a bug? Open an issue
- **Feature requests**: Have an idea? Start a discussion
- **Documentation**: Improve docs, fix typos, add examples
- **Code**: Fix bugs, add features, improve performance
- **Testing**: Add tests, improve coverage

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/dream-trainer.git
cd dream-trainer
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install in Development Mode

```bash
# Install with all development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

---

## Code Style

### Formatting

We use `ruff` for formatting and linting:

```bash
# Format code
ruff format .

# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Hints

Use type hints for all public APIs:

```python
# Good
def train(
    self,
    epochs: int,
    batch_size: int = 32,
) -> dict[str, float]:
    ...

# Avoid
def train(self, epochs, batch_size=32):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """Short description of the function.

    Longer description if needed, explaining the behavior
    and any important details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.

    Example:
        >>> my_function(1, "hello")
        True
    """
    pass
```

### Import Order

Imports should be organized:

```python
# Standard library
import os
from dataclasses import dataclass

# Third-party
import torch
import torch.nn as nn

# Local
from dream_trainer import BaseTrainer
from dream_trainer.callbacks import Callback
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_trainer.py

# Run with coverage
pytest --cov=dream_trainer

# Run only fast tests (no GPU required)
pytest -m "not slow"
```

### Writing Tests

```python
import pytest
import torch
from dream_trainer import BaseTrainer


class TestMyFeature:
    def test_basic_functionality(self):
        """Test the basic case."""
        # Arrange
        trainer = create_trainer()

        # Act
        result = trainer.my_feature()

        # Assert
        assert result == expected

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_with_various_inputs(self, input, expected):
        """Test with multiple input values."""
        result = my_function(input)
        assert result == expected

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_distributed_training(self):
        """Test that requires GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # Test distributed functionality
```

### Test Fixtures

```python
import pytest
from dream_trainer import DreamTrainerConfig


@pytest.fixture
def config():
    """Create a basic config for testing."""
    return DreamTrainerConfig(
        training_parameters=TrainingParameters(
            n_epochs=1,
            train_batch_size=2,
        ),
    )


@pytest.fixture
def trainer(config):
    """Create a trainer for testing."""
    return MyTrainer(config)
```

---

## Documentation

### Building Docs

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build docs
mkdocs build

# Serve locally
mkdocs serve
```

### Writing Documentation

- Use clear, simple language
- Include code examples
- Add cross-references to related pages
- Test all code examples

### Docstring to API Reference

Documentation is auto-generated from docstrings using mkdocstrings:

```python
class MyClass:
    """Short description.

    Longer description with details about the class.

    Attributes:
        attr1: Description of attr1.
        attr2: Description of attr2.

    Example:
        >>> obj = MyClass()
        >>> obj.method()
        result
    """
```

---

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines (`ruff format . && ruff check .`)
- [ ] All tests pass (`pytest`)
- [ ] Documentation updated if needed
- [ ] New features have tests
- [ ] Commit messages are clear

### 2. Create Pull Request

- Use a clear, descriptive title
- Reference related issues
- Describe what changed and why
- Include any migration steps if needed

### 3. PR Template

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

How has this been tested?

## Checklist

- [ ] Code formatted with ruff
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or migration guide provided)
```

### 4. Review Process

- Maintainers will review your PR
- Address any feedback
- Once approved, a maintainer will merge

---

## Adding New Features

### Adding a New Callback

1. Create the callback in `src/dream_trainer/callbacks/`:

```python
# src/dream_trainer/callbacks/my_callback.py
from dream_trainer.callbacks import Callback


class MyCallback(Callback):
    """Description of what this callback does.

    Args:
        param: Description.

    Example:
        >>> callback = MyCallback(param=value)
    """

    def __init__(self, param: int = 10):
        super().__init__()
        self.param = param

    def post_train_step(self, output: dict, batch_idx: int):
        # Implementation
        pass
```

2. Export in `__init__.py`:

```python
# src/dream_trainer/callbacks/__init__.py
from .my_callback import MyCallback

__all__ = [..., "MyCallback"]
```

3. Add tests:

```python
# tests/callbacks/test_my_callback.py
def test_my_callback():
    callback = MyCallback()
    # Test the callback
```

4. Add documentation:

```markdown
<!-- docs/api/callbacks/my_callback.md -->
# MyCallback

::: dream_trainer.callbacks.MyCallback
```

### Adding a New Mixin

1. Create the mixin in `src/dream_trainer/trainer/mixins/`:

```python
# src/dream_trainer/trainer/mixins/my_mixin.py
from dataclasses import dataclass


@dataclass
class MyMixinConfigMixin:
    """Configuration for MyMixin."""
    my_param: float = 1.0


class MyMixin:
    """Adds feature X to trainers.

    Example:
        >>> class MyTrainer(BaseTrainer, SetupMixin, MyMixin):
        ...     pass
    """

    config: MyMixinConfigMixin

    def my_method(self):
        """Description of method."""
        pass
```

2. Export and document as with callbacks.

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the [Python Code of Conduct](https://www.python.org/psf/conduct/)

---

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/dream3d/dream-trainer/discussions)
- **Bugs**: [GitHub Issues](https://github.com/dream3d/dream-trainer/issues)
- **Chat**: [Discord](https://discord.gg/dream-trainer)

---

## Recognition

Contributors are recognized in:
- The CONTRIBUTORS file
- Release notes
- The project README

Thank you for contributing to Dream Trainer!
