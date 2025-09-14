"""
Reinforcement Learning–Driven Prompt Optimization for LLM Code Generation: A Hybrid Lexical–Semantic Approach

A sophisticated machine learning framework that combines Proximal Policy Optimization (PPO) 
with Large Language Models (LLMs) for automated code generation and prompt optimization.
Implements a Hybrid Lexical–Semantic Approach for reinforcement learning-driven prompt optimization.
"""

__version__ = "1.0.0"
__author__ = ""
__email__ = ""

from .agents import PPOAgent
from .environments import RLPromptEnv
from .models import PromptRewriter, EpicGA
from .utils import Config, setup_logging
from .main import ReinforcementLearningPromptOptimizer

__all__ = [
    "PPOAgent",
    "RLPromptEnv", 
    "PromptRewriter",
    "EpicGA",
    "Config",
    "setup_logging",
    "ReinforcementLearningPromptOptimizer",
]
