"""
Reinforcement Learning Environment for prompt optimization.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import re
import time
import traceback
from typing import List, Optional, Tuple, Dict, Any, NamedTuple
from collections import namedtuple

from ..models import PromptRewriter, EpicGA, CodeT5Model
from ..utils.text_processing import CodeProcessor
from ..config import EnvironmentConfig, ModelConfig


class TestResult(NamedTuple):
    """Test result structure."""
    passed: bool
    error_message: str
    execution_time: float


class RLPromptEnv(gym.Env):
    """
    Reinforcement Learning Environment for prompt optimization.
    """
    
    def __init__(
        self, 
        dataset, 
        prompt_rewriter: PromptRewriter,
        epic_ga: EpicGA,
        codet5_model: CodeT5Model,
        config: EnvironmentConfig
    ):
        """
        Initialize RL environment.
        
        Args:
            dataset: Dataset for training
            prompt_rewriter: Prompt rewriter model
            epic_ga: Genetic algorithm for text mutation
            codet5_model: CodeT5+ model for code generation
            config: Environment configuration
        """
        super().__init__()
        
        self.dataset = dataset
        self.prompt_rewriter = prompt_rewriter
        self.epic_ga = epic_ga
        self.codet5_model = codet5_model
        self.config = config
        self.code_processor = CodeProcessor()
        
        # Environment state
        self.current_idx = 0
        self.current_step = 0
        self.current_prompt = ""
        self.changed_prompt = ""
        self.feedback_history = []
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # 0: keep, 1: rewrite, 2: mutate
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,), 
            dtype=np.float32
        )
    
    def _get_clean_prompt(self, idx: int) -> str:
        """
        Get clean prompt from dataset.
        
        Args:
            idx: Dataset index
            
        Returns:
            Clean prompt string
        """
        prompt = self.dataset[idx]["text"]
        return prompt[0] if isinstance(prompt, list) else str(prompt)
    
    def run_asserts(
        self, 
        code_str: str, 
        setup_code: str, 
        tests: List[str]
    ) -> List[TestResult]:
        """
        Run test assertions on generated code.
        
        Args:
            code_str: Generated code
            setup_code: Setup code
            tests: List of test cases
            
        Returns:
            List of test results
        """
        results = []
        
        def _short(s: str, limit: int = 120) -> str:
            return " ".join((s or "").strip().split())[:limit]
        
        for test in tests:
            start_time = time.time()
            try:
                # Create execution namespace
                namespace = {}
                
                # Execute setup code if provided
                if setup_code:
                    exec(setup_code, namespace)
                
                # Execute generated code
                exec(code_str, namespace)
                
                # Execute test
                exec(test, namespace)
                
                execution_time = time.time() - start_time
                results.append(TestResult(True, "", execution_time))
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = _short(str(e), 200)
                results.append(TestResult(False, error_msg, execution_time))
        
        return results
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Observation vector
        """
        self.current_prompt = self._get_clean_prompt(self.current_idx)
        
        if not self.current_prompt.strip():
            return np.zeros(10, dtype=np.float32)
        
        # Create observation features
        obs = np.zeros(10, dtype=np.float32)
        
        # Basic prompt features
        obs[0] = len(self.current_prompt) / 1000.0  # Normalized length
        obs[1] = len(self.current_prompt.split()) / 100.0  # Word count
        obs[2] = len(self.feedback_history) / 10.0  # Feedback count
        
        # Step features
        obs[3] = self.current_step / 10.0  # Current step
        obs[4] = self.current_idx / len(self.dataset)  # Progress
        
        # Prompt complexity features
        obs[5] = float("def " in self.current_prompt)  # Contains function definition
        obs[6] = float("class " in self.current_prompt)  # Contains class definition
        obs[7] = float("import " in self.current_prompt)  # Contains imports
        
        # Feedback features
        if self.feedback_history:
            obs[8] = len(self.feedback_history[-1]) / 1000.0  # Last feedback length
            obs[9] = float("error" in self.feedback_history[-1].lower())  # Contains error
        else:
            obs[8] = 0.0
            obs[9] = 0.0
        
        return obs
    
    def extract_function_name(self, assert_statement: str) -> Optional[str]:
        """
        Extract function name from assert statement.
        
        Args:
            assert_statement: Assert statement
            
        Returns:
            Function name or None
        """
        match = re.search(r"assert\s+(\w+)\s*\(", assert_statement)
        return match.group(1) if match else None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment.
        
        Args:
            seed: Random seed
            
        Returns:
            Initial observation and info
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_idx = 0
        self.feedback_history = []
        
        observation = self._get_observation()
        info = {
            "prompt_id": self.current_idx,
            "prompt": self.current_prompt,
            "step": self.current_step
        }
        
        return observation, info
    
    def create_meta_prompt(self, prompt: str, test_cases: List[str]) -> str:
        """
        Create meta prompt for code generation.
        
        Args:
            prompt: Original prompt
            test_cases: List of test cases
            
        Returns:
            Meta prompt
        """
        return self.codet5_model.create_meta_prompt(prompt, test_cases)
    
    def code_generating(self, test_cases: List[str]) -> str:
        """
        Generate code using CodeT5+ model.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Generated code
        """
        meta_prompt = self.create_meta_prompt(self.changed_prompt, test_cases)
        generated_texts = self.codet5_model.generate_code(meta_prompt)
        return generated_texts[0] if generated_texts else ""
    
    def format_feedback_for_llm(self, failures: List[TestResult], pass_mask: List[bool]) -> str:
        """
        Format feedback for LLM.
        
        Args:
            failures: List of failed test results
            pass_mask: List of pass/fail indicators
            
        Returns:
            Formatted feedback string
        """
        lines = [f"TEST_PASS_MASK: {tuple(pass_mask)}"]
        
        for failure in failures:
            if not failure.passed:
                lines.append(f"FAILED: {failure.error_message}")
        
        return "\n".join(lines)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0: keep, 1: rewrite, 2: mutate)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Get test cases for current prompt
        test_cases = self.dataset[self.current_idx]["test_list"]
        
        def _gen_and_eval() -> Tuple[bool, str, List[TestResult]]:
            """Generate code and evaluate it."""
            raw_code = self.code_generating(test_cases)
            code = self.code_processor.parse_code_block(raw_code, "python") or raw_code
            func_code = self.code_processor.parse_first_func(code)
            
            if not func_code:
                return False, "No function found in generated code", []
            
            # Run tests
            setup_code = self.dataset[self.current_idx].get("setup_code", "")
            test_results = self.run_asserts(func_code, setup_code, test_cases)
            
            # Check if all tests passed
            all_passed = all(result.passed for result in test_results)
            
            # Format feedback
            pass_mask = [result.passed for result in test_results]
            feedback = self.format_feedback_for_llm(test_results, pass_mask)
            
            return all_passed, feedback, test_results
        
        # Execute action
        if action == 0:  # Keep original prompt
            self.changed_prompt = self.current_prompt
        elif action == 1:  # Rewrite prompt
            function_name = self.extract_function_name(test_cases[0]) if test_cases else "function"
            feedback = self.feedback_history[-1] if self.feedback_history else ""
            self.changed_prompt = self.prompt_rewriter.modify(
                self.current_prompt, function_name, feedback
            )
        elif action == 2:  # Mutate prompt
            mutated_prompts = self.epic_ga.mutate_sentence(self.current_prompt)
            self.changed_prompt = mutated_prompts[0] if mutated_prompts else self.current_prompt
        
        # Generate and evaluate code
        is_passing, feedback, test_results = _gen_and_eval()
        self.feedback_history.append(feedback)
        
        # Calculate reward
        reward = 0.0
        if is_passing:
            reward = self.config.success_reward
        else:
            reward = self.config.penalty_scale
        
        # Scale reward
        reward *= self.config.reward_scale
        
        # Check termination conditions
        terminated = is_passing or self.current_step >= 10
        truncated = self.current_step >= 10
        
        # Move to next prompt if terminated
        if terminated:
            self.current_idx = (self.current_idx + 1) % len(self.dataset)
            self.current_step = 0
            self.feedback_history = []
        
        # Get next observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            "is_passing": is_passing,
            "prompt_id": self.current_idx,
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "feedback": feedback,
            "test_results": test_results,
            "changed_prompt": self.changed_prompt,
            "original_prompt": self.current_prompt
        }
        
        return observation, reward, terminated, truncated, info
    
    def get_env_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Environment information dictionary
        """
        return {
            "dataset_size": len(self.dataset),
            "current_idx": self.current_idx,
            "current_step": self.current_step,
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "config": self.config.__dict__
        }
