# Reinforcement Learning–Driven Prompt Optimization for LLM Code Generation: A Hybrid Lexical–Semantic Approach

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ppo-codet5.svg)](https://badge.fury.io/py/ppo-codet5)

A sophisticated machine learning framework that combines Proximal Policy Optimization (PPO) with Large Language Models (LLMs) for automated code generation and prompt optimization using the MBPP (Microsoft Python Programming Benchmark) dataset. This project implements a **Hybrid Lexical–Semantic Approach** for reinforcement learning-driven prompt optimization.

## 🚀 Overview

This project implements a reinforcement learning system that learns to generate better prompts for code generation tasks. The system supports multiple LLM backends and uses:

- **PPO (Proximal Policy Optimization)** for learning optimal prompt modification strategies
- **CodeT5+ (Salesforce/codet5p-770m-py)** for code generation (Notebook: `01_ppo_codet5.ipynb`)
- **CodeLlama-7b-Instruct** for code generation (Notebook: `02_PPO_LlaMa.ipynb`)
- **LLaMA-3.2-3B-Instruct** for prompt rewriting and optimization
- **MBPP Dataset** for training and evaluation (374 training samples, 974 total problems)
- **Epic_GA** ([EPiC Framework](https://github.com/HamedTaherkhani/EPiC)) for evolutionary prompt engineering and text mutation with semantic preservation

## 🏗️ Architecture

### Core Components

1. **RLPromptEnv**: A Gymnasium environment that simulates code generation tasks
2. **PPOAgent**: Implements the PPO algorithm for learning optimal actions
3. **PromptRewriter**: Uses LLaMA to rewrite prompts based on feedback
4. **Epic_GA**: Evolutionary prompt engineering algorithm (based on [EPiC Framework](https://github.com/HamedTaherkhani/EPiC)) for text mutation with semantic preservation
5. **CodeT5+ Integration**: Handles code generation and evaluation

### Workflow

1. **Environment Setup**: Load MBPP dataset and initialize models
2. **Prompt Processing**: Extract and clean prompts from the dataset
3. **Code Generation**: Use CodeT5+ to generate Python functions
4. **Evaluation**: Test generated code against provided test cases
5. **Reward Calculation**: Compute rewards based on test results
6. **PPO Training**: Update policy based on rewards and feedback
7. **Prompt Optimization**: Use LLaMA to rewrite prompts for better results

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **GPU**: NVIDIA A100 80GB PCIe (recommended for full training)
  - **Minimum**: 40GB VRAM for basic training
  - **Recommended**: 80GB VRAM for optimal performance
  - **Large-scale**: 80x NVIDIA A100 80GB PCIe for distributed training
- **RAM**: 32GB+ (64GB+ recommended)
- **Storage**: 50GB+ free space

### Python Dependencies
```bash
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
gymnasium>=0.26.0
numpy>=1.21.0
pandas>=1.3.0
nltk>=3.7
sentence-transformers>=2.2.0
gensim>=4.2.0
```

## 🛠️ Installation

### Quick Install

```bash
pip install ppo-codet5
```

### Development Install

1. **Clone the repository**:
   ```bash
   git clone https://github.com/promptoptrl/PROMIS.git
   cd PROMIS
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Install with development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('stopwords')
   ```

3. **Set up cache directories**:
   ```bash
   mkdir -p cache/hf cache/gensim cache/nltk
   ```

## 🚀 Usage

### Command Line Interface

```bash
# Create configuration file
ppo-codet5 config --output config.json

# Train the model
ppo-codet5 train --config config.json --output-dir ./results

# Evaluate trained model
ppo-codet5 eval --checkpoint ./results/final_model.pt --episodes 50
```

### Python API

```python
from ppo_codet5 import Config, PPOCodeT5Trainer

# Create configuration
config = Config()

# Initialize trainer
trainer = PPOCodeT5Trainer(config)

# Train
results = trainer.train("./outputs")

# Evaluate
eval_results = trainer.evaluate(num_episodes=10)
```

### Jupyter Notebooks

```python
# Run CodeT5+ based implementation
jupyter notebook 01_ppo_codet5.ipynb

# Run CodeLlama based implementation  
jupyter notebook 02_PPO_LlaMa.ipynb
```

### Key Configuration

The system can be configured through the `Config` class:

```python
from ppo_codet5 import Config

config = Config()

# Model configuration
config.model.codet5_model = "Salesforce/codet5p-770m-py"
config.model.temperature = 0.7
config.model.top_p = 0.9

# Training configuration
config.training.learning_rate = 3e-4
config.training.gamma = 0.99
config.training.clip_epsilon = 0.2
config.training.max_episodes = 1000  # Training episodes

# Environment configuration
config.environment.dataset_name = "mbpp"
config.environment.max_samples = None  # Use all 374 training samples
config.environment.reward_scale = 1.0

# Dataset information
# train_data = load_dataset("mbpp")["train"]  # 374 samples
# Last prompt_id in CSV: 974
```

### Training Parameters

- **Episodes**: Number of training episodes (default: 1000)
- **Steps per Episode**: Maximum steps per episode (default: 10)
- **Learning Rate**: PPO learning rate (default: 3e-4)
- **Gamma**: Discount factor (default: 0.99)
- **Clip Epsilon**: PPO clipping parameter (default: 0.2)

## 📊 Dataset Information

### MBPP (Microsoft Python Programming Benchmark)

- **Total Problems**: 974 programming problems
- **Training Split**: 374 samples (`train_data = load_dataset("mbpp")["train"]`)
- **Last prompt_id in CSV**: 974
- **Source**: [Microsoft Research MBPP Repository](https://github.com/microsoft/MBPP)
- **Format**: Python programming problems with test cases
- **Difficulty**: Beginner to intermediate level programming tasks

### Dataset Loading
```python
from datasets import load_dataset

# Load MBPP dataset
train_data = load_dataset("mbpp")["train"]  # 374 samples
print(f"MBPP train size: {len(train_data)}")  # Output: 374
```

## 📊 Results and Evaluation

The system outputs detailed results including:

- **Training Progress**: Episode rewards and success rates
- **Test Results**: CSV files with detailed test case results
- **Model Checkpoints**: Saved PPO agent weights
- **Performance Metrics**: Average rewards and success rates

### Output Files

- `01_Test/prompt_ENG_train_MBPP.csv`: Training results
- `ppo_agent_mbpp.pt`: Trained PPO model weights
- Console logs with detailed progress information

## 🔬 Technical Details

### PPO Implementation

The PPO agent uses a two-network architecture:
- **Actor Network**: Policy network for action selection
- **Critic Network**: Value network for state evaluation

### Code Generation Pipeline

1. **Prompt Extraction**: Clean and normalize prompts from MBPP
2. **Meta-Prompt Creation**: Build comprehensive prompts with test cases
3. **Code Generation**: Use CodeT5+ to generate Python functions
4. **Code Parsing**: Extract function definitions from generated code
5. **Test Execution**: Run test cases and collect results
6. **Feedback Processing**: Analyze failures and generate feedback

### Epic_GA (EPiC Framework) Features

The Epic_GA component is based on the [EPiC Framework](https://github.com/HamedTaherkhani/EPiC) by HamedTaherkhani, which implements:

- **Evolutionary Prompt Engineering**: Lightweight evolutionary algorithm for cost-effective code generation
- **Semantic Preservation**: Maintains meaning while mutating text through evolutionary operations
- **POS Tagging**: Uses NLTK for part-of-speech analysis and intelligent word substitution
- **WordNet Integration**: Leverages WordNet for semantic relationships and synonym discovery
- **FastText Vectors**: Uses pre-trained embeddings for similarity-based mutations
- **Cost-Effective Optimization**: Minimizes LLM calls while maximizing code quality improvements

## 📈 Performance

The system is designed to:
- Learn optimal prompt modification strategies
- Improve code generation success rates over time
- Handle complex programming tasks from MBPP dataset
- Provide detailed feedback and evaluation metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Salesforce](https://www.salesforce.com/)** for the [CodeT5+ model](https://huggingface.co/Salesforce/codet5p-770m-py)
- **[Microsoft Research](https://www.microsoft.com/en-us/research/)** for the [MBPP dataset](https://github.com/microsoft/MBPP)
- **[Meta AI](https://ai.meta.com/)** for the [LLaMA](https://github.com/facebookresearch/llama) and [CodeLlama](https://github.com/facebookresearch/codellama) models
- **[OpenAI](https://openai.com/)** for the [PPO algorithm](https://openai.com/blog/openai-baselines-ppo/)
- **[Hugging Face](https://huggingface.co/)** for the [Transformers library](https://github.com/huggingface/transformers)
- **[HamedTaherkhani/EPiC](https://github.com/HamedTaherkhani/EPiC)** for the Epic_GA (Evolutionary Prompt Engineering for Code) algorithm
- **EPiC Framework** for the lightweight evolutionary algorithm for cost-effective code generation

## 📞 Contact

For questions or support, please open an issue in the repository or contact the maintainers.

---

## ⚠️ Hardware Requirements

**Important**: This project requires significant computational resources:

- **Minimum**: 40GB VRAM for basic training
- **Recommended**: 80GB VRAM (NVIDIA A100 80GB PCIe)
- **Large-scale Training**: 80x NVIDIA A100 80GB PCIe for distributed training
- **Memory**: 32GB+ RAM (64GB+ recommended)
- **Storage**: 50GB+ free space

Ensure you have adequate GPU memory and processing power before running the training pipeline.
