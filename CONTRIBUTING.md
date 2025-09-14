# Contributing to PPO-CodeT5

Thank you for your interest in contributing to PPO-CodeT5! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended for training)

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/promptoptrl/PROMIS/ppo-codet5.git
   cd ppo-codet5
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## 📝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker
- Provide detailed information about the problem
- Include system information and error messages
- Use appropriate labels

### Suggesting Enhancements
- Open an issue with the "enhancement" label
- Describe the proposed feature clearly
- Explain the use case and benefits

### Code Contributions

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   pytest tests/
   ```

4. **Format your code**:
   ```bash
   black .
   flake8 .
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and small

### Jupyter Notebooks
- Clear cell outputs before committing
- Use descriptive markdown cells
- Organize code logically
- Include explanations for complex logic

### Documentation
- Update README.md for significant changes
- Add docstrings to new functions/classes
- Include examples in documentation

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ppo_codet5

# Run specific test file
pytest tests/test_specific.py
```

### Writing Tests
- Test both success and failure cases
- Use descriptive test names
- Mock external dependencies
- Aim for good test coverage

## 📊 Performance Considerations

### Memory Usage
- Monitor GPU memory usage during training
- Use appropriate batch sizes
- Clear unused variables and tensors

### Computational Efficiency
- Profile code for bottlenecks
- Use vectorized operations when possible
- Consider model quantization for deployment

## 🔒 Security

- Never commit API keys or credentials
- Use environment variables for sensitive data
- Validate all user inputs
- Keep dependencies updated

## 📞 Getting Help

- Check existing issues and discussions
- Join our community discussions
- Contact maintainers for urgent issues

## 🎯 Areas for Contribution

### High Priority
- Performance optimizations
- Additional model architectures
- Better evaluation metrics
- Documentation improvements

### Medium Priority
- New datasets support
- Visualization tools
- CLI improvements
- Docker support

### Low Priority
- Additional language support
- Web interface
- Mobile deployment
- Advanced hyperparameter tuning

## 📄 License

By contributing to PPO-CodeT5, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to PPO-CodeT5! 🎉
