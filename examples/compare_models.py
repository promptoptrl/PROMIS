"""
Compare CodeT5+ and CodeLlama implementations.
"""

import os
import sys
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo_codet5 import Config, ReinforcementLearningPromptOptimizer
from ppo_codet5.utils import setup_logging


def run_codet5_experiment():
    """Run CodeT5+ based experiment."""
    print("🚀 Starting CodeT5+ Experiment")
    
    # Create configuration for CodeT5+
    config = Config()
    config.model.codet5_model = "Salesforce/codet5p-770m-py"
    config.training.max_episodes = 20
    config.training.max_steps_per_episode = 5
    config.environment.max_samples = 50
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize trainer
    trainer = ReinforcementLearningPromptOptimizer(config)
    
    # Train
    start_time = time.time()
    results = trainer.train("./examples/outputs/codet5")
    training_time = time.time() - start_time
    
    # Evaluate
    eval_results = trainer.evaluate(num_episodes=5)
    
    return {
        "model": "CodeT5+",
        "training_time": training_time,
        "training_results": results,
        "evaluation_results": eval_results
    }


def run_llama_experiment():
    """Run CodeLlama based experiment."""
    print("🚀 Starting CodeLlama Experiment")
    
    # Create configuration for CodeLlama
    config = Config()
    config.model.codet5_model = "codellama/CodeLlama-7b-Instruct-hf"
    config.training.max_episodes = 20
    config.training.max_steps_per_episode = 5
    config.environment.max_samples = 50
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize trainer
    trainer = ReinforcementLearningPromptOptimizer(config)
    
    # Train
    start_time = time.time()
    results = trainer.train("./examples/outputs/llama")
    training_time = time.time() - start_time
    
    # Evaluate
    eval_results = trainer.evaluate(num_episodes=5)
    
    return {
        "model": "CodeLlama",
        "training_time": training_time,
        "training_results": results,
        "evaluation_results": eval_results
    }


def compare_results(codet5_results, llama_results):
    """Compare results from both models."""
    print("\n📊 Model Comparison Results:")
    print("=" * 50)
    
    # Training time comparison
    print(f"Training Time:")
    print(f"  CodeT5+: {codet5_results['training_time']:.2f} seconds")
    print(f"  CodeLlama: {llama_results['training_time']:.2f} seconds")
    
    # Performance comparison
    codet5_avg_reward = codet5_results['evaluation_results']['avg_reward']
    llama_avg_reward = llama_results['evaluation_results']['avg_reward']
    
    codet5_success_rate = codet5_results['evaluation_results']['success_rate']
    llama_success_rate = llama_results['evaluation_results']['success_rate']
    
    print(f"\nAverage Reward:")
    print(f"  CodeT5+: {codet5_avg_reward:.3f}")
    print(f"  CodeLlama: {llama_avg_reward:.3f}")
    
    print(f"\nSuccess Rate:")
    print(f"  CodeT5+: {codet5_success_rate:.3f}")
    print(f"  CodeLlama: {llama_success_rate:.3f}")
    
    # Determine winner
    if codet5_avg_reward > llama_avg_reward:
        print(f"\n🏆 CodeT5+ performs better in terms of average reward")
    elif llama_avg_reward > codet5_avg_reward:
        print(f"\n🏆 CodeLlama performs better in terms of average reward")
    else:
        print(f"\n🤝 Both models perform equally well")
    
    if codet5_success_rate > llama_success_rate:
        print(f"🏆 CodeT5+ has higher success rate")
    elif llama_success_rate > codet5_success_rate:
        print(f"🏆 CodeLlama has higher success rate")
    else:
        print(f"🤝 Both models have equal success rates")


def main():
    """Main comparison function."""
    print("🔬 Reinforcement Learning–Driven Prompt Optimization Model Comparison Experiment")
    print("Comparing CodeT5+ vs CodeLlama implementations")
    print("=" * 60)
    
    # Run experiments
    codet5_results = run_codet5_experiment()
    llama_results = run_llama_experiment()
    
    # Compare results
    compare_results(codet5_results, llama_results)
    
    # Save comparison results
    comparison_data = {
        "codet5_results": codet5_results,
        "llama_results": llama_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_path = "./examples/outputs/model_comparison.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Comparison results saved to {results_path}")
    print("\n✅ Model comparison completed successfully!")


if __name__ == "__main__":
    main()
