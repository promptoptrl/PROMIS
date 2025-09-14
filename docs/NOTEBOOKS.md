# Reinforcement Learning–Driven Prompt Optimization for LLM Code Generation: Notebooks Documentation

This document provides detailed information about the Jupyter notebooks included in this reinforcement learning-driven prompt optimization project.

## 📓 Available Notebooks

### 1. `01_ppo_codet5.ipynb` - CodeT5+ Implementation

**Purpose**: Implements PPO-based prompt optimization using CodeT5+ for code generation.

**Key Features**:
- Uses **Salesforce/codet5p-770m-py** model for code generation
- Implements PPO agent for learning optimal prompt modifications
- Includes Epic_GA (based on [EPiC Framework](https://github.com/HamedTaherkhani/EPiC)) for evolutionary prompt engineering
- Uses LLaMA-3.2-3B-Instruct for prompt rewriting
- Trains on MBPP dataset

**Architecture**:
```
Input Prompt → Prompt Rewriter (LLaMA) → CodeT5+ → Generated Code → Evaluation → PPO Update
```

**Configuration**:
```python
class Config:
    MODEL_NAME = "Salesforce/codet5p-770m-py"
    EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    MAX_STEPS = 10
```

### 2. `02_PPO_LlaMa.ipynb` - CodeLlama Implementation

**Purpose**: Implements PPO-based prompt optimization using CodeLlama for code generation.

**Key Features**:
- Uses **codellama/CodeLlama-7b-Instruct-hf** model for code generation
- Implements PPO agent for learning optimal prompt modifications
- Includes Epic_GA (based on [EPiC Framework](https://github.com/HamedTaherkhani/EPiC)) for evolutionary prompt engineering
- Uses LLaMA-3.2-3B-Instruct for prompt rewriting
- Trains on MBPP dataset

**Architecture**:
```
Input Prompt → Prompt Rewriter (LLaMA) → CodeLlama → Generated Code → Evaluation → PPO Update
```

**Configuration**:
```python
class Config:
    MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
    EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    MAX_STEPS = 10
```

## 🔄 Key Differences

| Aspect | CodeT5+ (01) | CodeLlama (02) |
|--------|--------------|----------------|
| **Code Generation Model** | Salesforce/codet5p-770m-py | codellama/CodeLlama-7b-Instruct-hf |
| **Model Size** | 770M parameters | 7B parameters |
| **Specialization** | Code-specific T5 | General-purpose LLM |
| **Performance** | Faster inference | More capable generation |
| **Memory Usage** | Lower | Higher |
| **Code Quality** | Good for simple tasks | Better for complex tasks |

## 🚀 Usage Instructions

### Running Individual Notebooks

1. **CodeT5+ Implementation**:
   ```bash
   jupyter notebook 01_ppo_codet5.ipynb
   ```

2. **CodeLlama Implementation**:
   ```bash
   jupyter notebook 02_PPO_LlaMa.ipynb
   ```

### Running Both for Comparison

Use the comparison script:
```bash
python examples/compare_models.py
```

## 📊 Expected Results

### CodeT5+ Results
- **Training Time**: ~30-60 minutes (depending on hardware)
- **Success Rate**: 60-80% on MBPP dataset
- **Memory Usage**: ~4-6GB GPU memory
- **Best For**: Quick prototyping, simple code generation tasks

### CodeLlama Results
- **Training Time**: ~60-120 minutes (depending on hardware)
- **Success Rate**: 70-90% on MBPP dataset
- **Memory Usage**: ~12-16GB GPU memory
- **Best For**: Complex code generation, production use

## 🔧 Customization

### Modifying Model Parameters

Both notebooks support customization through the `Config` class:

```python
# Modify training parameters
config.training.max_episodes = 100
config.training.learning_rate = 1e-4

# Modify environment parameters
config.environment.max_samples = 200
config.environment.reward_scale = 2.0

# Modify model parameters
config.model.temperature = 0.8
config.model.top_p = 0.95
```

### Adding New Models

To add support for new models:

1. Update the `Config` class with new model name
2. Modify the model initialization code
3. Adjust the code generation pipeline if needed
4. Update the environment to handle new model outputs

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use smaller model variants
   - Enable gradient checkpointing

2. **Slow Training**:
   - Reduce dataset size for testing
   - Use fewer episodes
   - Optimize model loading

3. **Poor Performance**:
   - Check model compatibility
   - Verify dataset format
   - Adjust hyperparameters

### Performance Optimization

1. **Memory Optimization**:
   ```python
   # Use mixed precision training
   config.model.torch_dtype = "float16"
   
   # Enable gradient checkpointing
   config.training.gradient_checkpointing = True
   ```

2. **Speed Optimization**:
   ```python
   # Reduce model size for testing
   config.environment.max_samples = 50
   
   # Use fewer training episodes
   config.training.max_episodes = 20
   ```

## 📈 Monitoring and Logging

Both notebooks include comprehensive logging:

- **Training Progress**: Episode rewards and success rates
- **Model Performance**: Code generation quality metrics
- **Resource Usage**: GPU memory and training time
- **Error Tracking**: Failed test cases and error analysis

## 🤝 Contributing

When contributing to the notebooks:

1. **Follow the existing structure**
2. **Add comprehensive comments**
3. **Include error handling**
4. **Test with different configurations**
5. **Update documentation**

## 🔬 EPiC Framework Integration

Both notebooks integrate the **Epic_GA** component, which is based on the [EPiC Framework](https://github.com/HamedTaherkhani/EPiC) by HamedTaherkhani. This framework implements:

### Evolutionary Prompt Engineering
- **Lightweight Algorithm**: Cost-effective evolutionary approach for prompt optimization
- **Semantic Preservation**: Maintains meaning while evolving prompts
- **Multi-Modal Mutation**: Supports both local (NLTK/Gensim) and LLM-based mutations

### Key EPiC Features Used
1. **WordNet Integration**: Semantic relationships for intelligent word substitution
2. **FastText Vectors**: Pre-trained embeddings for similarity-based mutations
3. **POS Tagging**: Part-of-speech analysis for context-aware mutations
4. **Cost Optimization**: Minimizes LLM calls while maximizing quality improvements

### EPiC Algorithm Flow
```
Initial Prompt → Code Generation → Test Evaluation → 
If Failed: Epic_GA Mutation → New Prompt → Repeat
```

This integration allows PROMIS to benefit from EPiC's proven evolutionary prompt engineering capabilities while adding the reinforcement learning layer for more sophisticated optimization.

## 📚 References

- [CodeT5+ Paper](https://arxiv.org/abs/2207.10397)
- [CodeLlama Paper](https://arxiv.org/abs/2308.12950)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [MBPP Dataset](https://arxiv.org/abs/2108.07732)
- [EPiC Framework](https://github.com/HamedTaherkhani/EPiC) - Evolutionary Prompt Engineering for Cost-Effective Code Generation
