# Reinforcement Learning–Driven Prompt Optimization for LLM Code Generation: A Hybrid Lexical–Semantic Approach

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ppo-codet5.svg)](https://badge.fury.io/py/ppo-codet5)


##  Overview

This project implements a reinforcement learning system that learns to generate better prompts for code generation tasks. The system supports multiple LLM backends and uses:

- **PPO (Proximal Policy Optimization)** for learning optimal prompt modification strategies
- **CodeT5+ (Salesforce/codet5p-770m-py)** for code generation (Notebook: `01_ppo_codet5.ipynb`)
- **CodeLlama-7b-Instruct** for code generation (Notebook: `02_PPO_LlaMa.ipynb`)
- **LLaMA-3.2-3B-Instruct** for prompt rewriting and optimization
- **MBPP Dataset** for training and evaluation (374 training samples, 974 total problems)
- **Epic_GA** ([EPiC Framework](https://github.com/HamedTaherkhani/EPiC)) for evolutionary prompt engineering and text mutation with semantic preservation
- **Reflexion-Inspired Feedback** ([Reflexion Framework](https://github.com/noahshinn/reflexion)) for verbal reinforcement learning and iterative prompt improvement

## Architecture

### Core Components

1. **RLPromptEnv**: A Gymnasium environment that simulates code generation tasks
2. **PPOAgent**: Implements the PPO algorithm for learning optimal actions
3. **PromptRewriter**: Uses LLaMA to rewrite prompts based on feedback
4. **Epic_GA**: Evolutionary prompt engineering algorithm (based on [EPiC Framework](https://github.com/HamedTaherkhani/EPiC)) for text mutation with semantic preservation
5. **Reflexion Integration**: Verbal reinforcement learning for feedback storage and prompt improvement
6. **Code Generation Models**: Handles code generation and evaluation

### Workflow

1. **Environment Setup**: Load MBPP dataset and initialize models
2. **Prompt Processing**: Extract and clean prompts from the dataset
3. **Code Generation**: Use CodeT5+ to generate Python functions
4. **Evaluation**: Test generated code against provided test cases
5. **Reward Calculation**: Compute rewards based on test results
6. **PPO Training**: Update policy based on rewards and feedback
7. **Prompt Optimization**: Use LLaMA to rewrite prompts for better results

##  Requirements

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

##  Installation

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

## Usage

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

##  Dataset Information

### MBPP (Google Python Programming Benchmark)

- **Total Problems**: 974 programming problems
- **Training Split**: 374 samples (`train_data = load_dataset("mbpp")["train"]`)
- **Last prompt_id in CSV**: 974
- **Source**: [Google Research MBPP Repository](https://github.com/google-research/google-research/tree/master/mbpp)
- **Format**: Python programming problems with test cases
- **Difficulty**: Beginner to intermediate level programming tasks

### Dataset Loading
```python
from datasets import load_dataset

# Load MBPP dataset
train_data = load_dataset("mbpp")["train"]  # 374 samples
print(f"MBPP train size: {len(train_data)}")  # Output: 374
```

##  Results and Evaluation

###  Performance Comparison on MBPP Benchmark

Our reinforcement learning-driven approach achieves state-of-the-art performance on the Google Python Programming Benchmark (MBPP), significantly outperforming existing methods including EPiC, Reflexion, and other baseline strategies.

| **Strategy** | **CodeT5+ Pass@1** | **CodeT5+ SoftPass@1** | **CodeLLaMA Pass@1** | **CodeLLaMA SoftPass@1** |
|--------------|-------------------|----------------------|---------------------|------------------------|
| **Direct Generation (Action 0)** | 12.84% | 22.80% | 41.91% | 48.82% |
| **Genetic Mutation (Action 1)** | 20.23% | 33.40% | 43.01% | 51.70% |
| **Semantic Rewriting (Action 2)** | 23.73% | 36.90% | 46.20% | 55.20% |
| **Randomized Selection (10 Steps)** | 31.12% | 44.10% | 49.08% | 58.60% |
| **EPiC** | 41.89% | 54.20% | 51.40% | 61.50% |
| **Reflexion** | 41.63% | 55.10% | 52.70% | 63.60% |
| **RL Agent (PPO, Ours)** | **57.58%** | **67.90%** | **64.80%** | **73.10%** |





##  Technical Details



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

### Reflexion Integration Features

The system incorporates concepts from the **Reflexion: Language Agents with Verbal Reinforcement Learning** framework (NeurIPS 2023), implementing:

- **Verbal Feedback**: Agents receive linguistic feedback on task performance
- **Episodic Memory**: Store reflections for future decision-making
- **Reflection Mechanism**: Agents verbally reflect on task feedback using LLaMA
- **Iterative Improvement**: Use past reflections to improve future performance
- **Multi-Modal Feedback**: Combine test results, semantic feedback, and evolutionary mutations

##  Performance

The system is designed to:
- Learn optimal prompt modification strategies
- Improve code generation success rates over time
- Handle complex programming tasks from MBPP dataset
- Provide detailed feedback and evaluation metrics

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **[Salesforce](https://www.salesforce.com/)** for the [CodeT5+ model](https://huggingface.co/Salesforce/codet5p-770m-py)
- **[Google Research](https://github.com/google-research/google-research/)** for the [MBPP dataset](https://github.com/google-research/google-research/tree/master/mbpp)
- **[Meta AI](https://ai.meta.com/)** for the [LLaMA](https://github.com/facebookresearch/llama) and [CodeLlama](https://github.com/facebookresearch/codellama) models
- **[OpenAI](https://openai.com/)** for the [PPO algorithm](https://openai.com/blog/openai-baselines-ppo/)
- **[Hugging Face](https://huggingface.co/)** for the [Transformers library](https://github.com/huggingface/transformers)
- **[HamedTaherkhani/EPiC](https://github.com/HamedTaherkhani/EPiC)** for the Epic_GA (Evolutionary Prompt Engineering for Code) algorithm
- **EPiC Framework** for the lightweight evolutionary algorithm for cost-effective code generation
- **[noahshinn/reflexion](https://github.com/noahshinn/reflexion)** for the Reflexion framework and verbal reinforcement learning concepts
- **Reflexion Framework** for language agents with verbal reinforcement learning (NeurIPS 2023)

---

##  Hardware Requirements

**Important**: This project requires significant computational resources:

- **Minimum**: 40GB VRAM for basic training
- **Recommended**: 80GB VRAM (NVIDIA A100 80GB PCIe)
- **Large-scale Training**: 80x NVIDIA A100 80GB PCIe for distributed training
- **Memory**: 32GB+ RAM (64GB+ recommended)
- **Storage**: 50GB+ free space

Ensure you have adequate GPU memory and processing power before running the training pipeline.
