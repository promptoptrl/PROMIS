from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "PPO-CodeT5: Reinforcement Learning for Code Generation"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

setup(
    name="ppo-codet5",
    version="1.0.0",
    author="",
    author_email="",
    description="Reinforcement Learning–Driven Prompt Optimization for LLM Code Generation: A Hybrid Lexical–Semantic Approach",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/promptoptrl/PROMIS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
            "torchvision>=0.10.0+cu111",
        ],
    },
    entry_points={
        "console_scripts": [
            "ppo-codet5=ppo_codet5.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ppo_codet5": ["*.json", "*.yaml", "*.yml"],
    },
)
