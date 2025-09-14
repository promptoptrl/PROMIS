# Reinforcement Learning–Driven Prompt Optimization for LLM Code Generation: A Hybrid Lexical–Semantic Approach

## 🎯 Project Overview

This project is a sophisticated machine learning framework that implements a **Hybrid Lexical–Semantic Approach** for reinforcement learning-driven prompt optimization in LLM code generation.

### 📊 Dataset Information
- **MBPP Dataset**: 374 training samples from 974 total problems
- **Training Command**: `train_data = load_dataset("mbpp")["train"]`
- **Last prompt_id**: 974
- **Source**: [Microsoft Research MBPP](https://github.com/microsoft/MBPP)

### 🖥️ Hardware Requirements
- **Minimum**: 40GB VRAM for basic training
- **Recommended**: 80GB VRAM (NVIDIA A100 80GB PCIe)
- **Large-scale**: 80x NVIDIA A100 80GB PCIe for distributed training

## 🏗️ Architecture

The project combines multiple cutting-edge technologies with proper attribution:

- **[Salesforce](https://www.salesforce.com/)** - CodeT5+ model
- **[Microsoft Research](https://www.microsoft.com/en-us/research/)** - MBPP dataset  
- **[Meta AI](https://ai.meta.com/)** - LLaMA and CodeLlama models
- **[OpenAI](https://openai.com/)** - PPO algorithm
- **[Hugging Face](https://huggingface.co/)** - Transformers library
- **[EPiC Framework](https://github.com/HamedTaherkhani/EPiC)** - Epic_GA component
- **[Reflexion Framework](https://github.com/noahshinn/reflexion)** - Verbal reinforcement learning concepts

### Core Technologies
- **PPO (Proximal Policy Optimization)**: Reinforcement learning algorithm for learning optimal prompt modification strategies
- **Large Language Models**: Support for both CodeT5+ and CodeLlama for code generation
- **Epic_GA ([EPiC Framework](https://github.com/HamedTaherkhani/EPiC))**: Evolutionary prompt engineering for cost-effective text mutation with semantic preservation
- **Reflexion-Inspired Feedback**: Verbal reinforcement learning concepts for iterative prompt improvement
- **MBPP Dataset**: Microsoft Python Programming Benchmark for training and evaluation

## 🔄 Reflexion Framework Integration

This project incorporates concepts from the **Reflexion: Language Agents with Verbal Reinforcement Learning** framework (NeurIPS 2023) to enhance prompt optimization through verbal feedback mechanisms.

### Reflexion Framework Overview

| **Aspect** | **Details** |
|------------|-------------|
| **Paper Title** | Reflexion: Language Agents with Verbal Reinforcement Learning |
| **Conference** | NeurIPS 2023 |
| **Authors** | Noah Shinn, Beck Labash, Ashwin Gopinath |
| **ArXiv ID** | 2303.11366 |
| **GitHub Repository** | [noahshinn/reflexion](https://github.com/noahshinn/reflexion) |
| **Key Innovation** | Verbal reinforcement learning through linguistic feedback instead of weight updates |

### Core Reflexion Concepts

| **Concept** | **Description** | **Implementation in This Project** |
|-------------|-----------------|-----------------------------------|
| **Verbal Feedback** | Agents receive linguistic feedback on task performance | ✅ Test execution feedback stored in `feedback_history` |
| **Episodic Memory** | Store reflections for future decision-making | ✅ Feedback persistence across episodes via `self.feedback` |
| **Reflection Mechanism** | Agents verbally reflect on task feedback | ✅ LLaMA-based prompt rewriting using previous feedback |
| **Iterative Improvement** | Use past reflections to improve future performance | ✅ PPO agent learns from feedback patterns over time |

### Reflexion Performance Results

| **Benchmark** | **Reflexion Result** | **Previous SOTA** | **Improvement** |
|---------------|---------------------|-------------------|-----------------|
| **HumanEval (Pass@1)** | 91% | GPT-4: 80% | +11% |
| **Sequential Decision-Making** | Significant improvements | Baseline methods | Substantial gains |
| **Language Reasoning** | Enhanced performance | Standard approaches | Notable improvements |

### Integration in Our Project

| **Component** | **Reflexion-Inspired Feature** | **Implementation Details** |
|---------------|-------------------------------|----------------------------|
| **Feedback Storage** | Episodic memory for reflections | `self.feedback_history.append(feedback)` |
| **Verbal Reflection** | Linguistic analysis of failures | LLaMA-3.2-3B-Instruct prompt rewriting |
| **Iterative Learning** | Use past feedback for improvement | PPO agent learns optimal prompt modification strategies |
| **Multi-Modal Feedback** | Combine different feedback types | Test results + semantic feedback + evolutionary mutations |

### Reflexion-Inspired Workflow

```
Initial Prompt → Code Generation → Test Evaluation → 
Verbal Feedback → Reflection Storage → Prompt Modification → 
Improved Generation → Enhanced Performance
```

This integration allows our reinforcement learning-driven prompt optimization system to benefit from Reflexion's proven verbal reinforcement learning capabilities, creating a hybrid approach that combines:
- **PPO-based optimization** for strategic prompt modification
- **Reflexion-inspired feedback** for linguistic reflection and improvement
- **EPiC evolutionary methods** for semantic-preserving mutations

## 📊 Experimental Results

### Performance Comparison on MBPP Benchmark

The following table shows a comprehensive comparison of Pass@1 (strict success rate) and SoftPass@1 (partial success, i.e., fraction of unit tests passed) across different prompt optimization strategies on the MBPP benchmark. Results are reported for both CodeT5+ and CodeLLaMA models. SoftPass@1 highlights incremental improvements missed by binary Pass@1, showing the benefits of lexical and semantic transformations as well as multi-step reinforcement learning with shaped rewards.

| **Strategy** | **CodeT5+ Pass@1** | **CodeT5+ SoftPass@1** | **CodeLLaMA Pass@1** | **CodeLLaMA SoftPass@1** |
|--------------|-------------------|----------------------|---------------------|------------------------|
| **Direct Generation (Action 0)** | 12.84% | 22.80% | 41.91% | 48.82% |
| **Genetic Mutation (Action 1)** | 20.23% | 33.40% | 43.01% | 51.70% |
| **Semantic Rewriting (Action 2)** | 23.73% | 36.90% | 46.20% | 55.20% |
| **Randomized Selection (10 Steps)** | 31.12% | 44.10% | 49.08% | 58.60% |
| **EPiC** | 41.89% | 54.20% | 51.40% | 61.50% |
| **Reflexion** | 41.63% | 55.10% | 52.70% | 63.60% |
| **🎯 RL Agent (PPO, Ours)** | **57.58%** | **67.90%** | **64.80%** | **73.10%** |

### Key Findings

#### 🏆 Superior Performance
- **Our RL Agent (PPO)** achieves the highest performance across all metrics
- **CodeT5+**: 57.58% Pass@1 (vs 41.89% EPiC, 41.63% Reflexion)
- **CodeLLaMA**: 64.80% Pass@1 (vs 52.70% Reflexion, 51.40% EPiC)

#### 📈 Significant Improvements
- **CodeT5+ Pass@1**: +15.69% improvement over EPiC, +15.95% over Reflexion
- **CodeT5+ SoftPass@1**: +13.70% improvement over EPiC, +12.80% over Reflexion
- **CodeLLaMA Pass@1**: +12.10% improvement over Reflexion, +13.40% over EPiC
- **CodeLLaMA SoftPass@1**: +9.50% improvement over Reflexion, +11.60% over EPiC

#### 🔍 Strategy Analysis
1. **Direct Generation**: Baseline performance showing room for improvement
2. **Genetic Mutation**: Modest improvements through evolutionary operations
3. **Semantic Rewriting**: Better results through LLM-based prompt modification
4. **Randomized Selection**: Multi-step approach shows benefits
5. **EPiC**: Strong evolutionary prompt engineering baseline
6. **Reflexion**: Competitive verbal reinforcement learning approach
7. **Our RL Agent**: Best performance through learned prompt optimization strategies

#### 💡 SoftPass@1 Insights
- **Incremental Progress**: SoftPass@1 reveals improvements missed by binary Pass@1
- **Lexical Benefits**: Shows value of semantic transformations
- **Multi-step Learning**: Demonstrates benefits of reinforcement learning with shaped rewards
- **Consistent Gains**: Our approach shows consistent improvements in both metrics

### Dual Implementation Approach

1. **CodeT5+ Implementation** (`01_ppo_codet5.ipynb`)
   - Model: Salesforce/codet5p-770m-py
   - Focus: Code-specific generation
   - Best for: Quick prototyping and simple tasks

2. **CodeLlama Implementation** (`02_PPO_LlaMa.ipynb`)
   - Model: codellama/CodeLlama-7b-Instruct-hf
   - Focus: General-purpose code generation
   - Best for: Complex tasks and production use

## 📁 Project Structure

```
Reinforcement-Learning-Driven-Prompt-Optimization/
├── ppo_codet5/                    # Professional Python package
│   ├── config.py                 # Configuration management
│   ├── main.py                   # Training/evaluation logic
│   ├── cli.py                    # Command line interface
│   ├── agents/                   # PPO agent implementation
│   ├── environments/             # RL environment
│   ├── models/                   # ML models (LLaMA, CodeT5+, GA)
│   └── utils/                    # Utility modules
├── examples/                     # Example scripts
│   ├── basic_training.py         # Basic training example
│   ├── custom_config.py          # Custom configuration example
│   ├── evaluation_example.py     # Evaluation example
│   └── compare_models.py         # Model comparison script
├── docs/                         # Documentation
│   └── NOTEBOOKS.md              # Notebook documentation
├── 01_ppo_codet5.ipynb          # CodeT5+ implementation
├── 02_PPO_LlaMa.ipynb           # CodeLlama implementation
├── README.md                     # Main documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── install scripts               # Cross-platform installation
```

## 🚀 Key Features

### Professional Package Structure
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Configuration Management**: Flexible, dataclass-based configuration
- ✅ **CLI Interface**: Professional command-line tools
- ✅ **Python API**: Clean programmatic interface
- ✅ **Comprehensive Documentation**: Extensive docs and examples

### Multiple Usage Options
- ✅ **Jupyter Notebooks**: Interactive development and experimentation
- ✅ **Command Line**: `ppo-codet5 config/train/eval` commands
- ✅ **Python API**: `from ppo_codet5 import Config, PPOCodeT5Trainer`
- ✅ **Comparison Tools**: Side-by-side model evaluation

### Advanced Capabilities
- ✅ **Hybrid Approach**: Combines lexical and semantic optimization
- ✅ **Multi-Model Support**: CodeT5+ and CodeLlama implementations
- ✅ **Genetic Algorithm**: Text mutation with semantic preservation
- ✅ **Comprehensive Evaluation**: Detailed metrics and analysis
- ✅ **Production Ready**: Scalable and maintainable codebase

## 🎯 Research Contributions

### Novel Approach
- **Hybrid Lexical–Semantic Optimization**: Combines traditional NLP techniques with modern LLM capabilities
- **Multi-Model Comparison**: Systematic evaluation of different LLM architectures
- **Reinforcement Learning Integration**: PPO-driven prompt optimization
- **Semantic Preservation**: Genetic algorithm maintains meaning while optimizing text

### Technical Innovations
- **Adaptive Prompt Rewriting**: LLaMA-based prompt optimization
- **Evolutionary Prompt Engineering**: Integration of EPiC Framework for cost-effective prompt evolution
- **Code-Specific Evaluation**: MBPP dataset integration
- **Multi-Agent Architecture**: PPO agent with specialized environments
- **Comprehensive Metrics**: Success rates, execution time, error analysis

## 🤝 Collaboration Ready

### Team Development Features
- ✅ **Professional Structure**: Industry-standard Python package
- ✅ **Clear Documentation**: Comprehensive guides and examples
- ✅ **Modular Design**: Easy for multiple developers to work on
- ✅ **Version Control**: Git-ready with proper .gitignore
- ✅ **Testing Framework**: Built-in testing capabilities
- ✅ **CI/CD Ready**: Automated testing and deployment support

### GitHub Integration
- ✅ **Repository**: https://github.com/promptoptrl/PROMIS
- ✅ **MIT License**: Open source and collaborative
- ✅ **Contributing Guidelines**: Clear contribution process
- ✅ **Issue Tracking**: GitHub issues for bug reports and features
- ✅ **Pull Request Workflow**: Standard GitHub collaboration

## 🎉 Ready for Production

The PROMIS project is now ready for:
- ✅ **Academic Research**: Publication-ready implementation
- ✅ **Industry Application**: Production-grade codebase
- ✅ **Team Collaboration**: Multi-developer workflow support
- ✅ **Open Source**: Community contribution and adoption
- ✅ **Further Development**: Extensible and maintainable architecture

## 📞 Next Steps

1. **Upload to GitHub**: Push to https://github.com/promptoptrl/PROMIS
2. **Team Onboarding**: Share repository with collaborators
3. **Documentation Review**: Ensure all docs are up-to-date
4. **Testing**: Run comprehensive tests on different environments
5. **Publication**: Prepare for academic or industry presentation

---

This project represents a significant advancement in the field of reinforcement learning-driven prompt optimization, combining cutting-edge LLM technology with sophisticated optimization techniques in a professional, collaborative framework.
