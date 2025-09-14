"""
Configuration management for PPO-CodeT5.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    
    # Model names
    codet5_model: str = "Salesforce/codet5p-770m-py"
    embedder_model: str = "all-MiniLM-L6-v2"
    llama_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Model parameters
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    torch_dtype: str = "float16"
    
    # Device configuration
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    def __post_init__(self):
        """Set device after initialization."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # PPO parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training parameters
    max_episodes: int = 1000
    max_steps_per_episode: int = 10
    batch_size: int = 64
    buffer_size: int = 10000
    
    # Evaluation parameters
    eval_episodes: int = 10
    eval_frequency: int = 100
    save_frequency: int = 500


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    
    # MBPP dataset information
    dataset_name: str = "mbpp"
    dataset_split: str = "train"
    total_problems: int = 974
    training_samples: int = 374
    last_prompt_id: int = 974
    
    # Dataset loading
    load_command: str = 'load_dataset("mbpp")["train"]'
    
    # Dataset source
    source_url: str = "https://github.com/microsoft/MBPP"
    source_organization: str = "Microsoft Research"


@dataclass
class EnvironmentConfig:
    """Configuration for environment parameters."""
    
    # Dataset configuration
    dataset_name: str = "mbpp"
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    
    # Environment parameters
    reward_scale: float = 1.0
    penalty_scale: float = -0.1
    success_reward: float = 1.0
    
    # Code generation parameters
    max_code_length: int = 512
    timeout_seconds: int = 30


@dataclass
class CacheConfig:
    """Configuration for cache directories."""
    
    # Cache directories
    transformers_cache: str = "./cache/hf"
    gensim_cache: str = "./cache/gensim"
    nltk_cache: str = "./cache/nltk"
    
    # Output directories
    output_dir: str = "./outputs"
    results_dir: str = "./results"
    checkpoints_dir: str = "./checkpoints"
    
    def __post_init__(self):
        """Create cache directories."""
        for cache_dir in [self.transformers_cache, self.gensim_cache, self.nltk_cache,
                         self.output_dir, self.results_dir, self.checkpoints_dir]:
            os.makedirs(cache_dir, exist_ok=True)


@dataclass
class Config:
    """Main configuration class combining all configurations."""
    
    model: ModelConfig = None
    training: TrainingConfig = None
    environment: EnvironmentConfig = None
    dataset: DatasetConfig = None
    cache: CacheConfig = None
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.dataset is None:
            self.dataset = DatasetConfig()
        if self.cache is None:
            self.cache = CacheConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "environment": self.environment.__dict__,
            "dataset": self.dataset.__dict__,
            "cache": self.cache.__dict__,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()
        
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "environment" in config_dict:
            config.environment = EnvironmentConfig(**config_dict["environment"])
        if "dataset" in config_dict:
            config.dataset = DatasetConfig(**config_dict["dataset"])
        if "cache" in config_dict:
            config.cache = CacheConfig(**config_dict["cache"])
        
        config.log_level = config_dict.get("log_level", "INFO")
        config.log_file = config_dict.get("log_file")
        config.seed = config_dict.get("seed", 42)
        
        return config
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
