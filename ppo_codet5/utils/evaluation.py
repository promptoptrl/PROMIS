"""
Evaluation utilities for PPO-CodeT5.
"""

import time
import ast
import traceback
from typing import List, Dict, Any, Tuple, Optional
from collections import namedtuple

TestResult = namedtuple('TestResult', ['passed', 'error_message', 'execution_time'])


class CodeEvaluator:
    """
    Code evaluation utilities.
    """
    
    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize code evaluator.
        
        Args:
            timeout_seconds: Timeout for code execution
        """
        self.timeout_seconds = timeout_seconds
    
    def run_tests(
        self, 
        code: str, 
        test_cases: List[str], 
        setup_code: str = ""
    ) -> List[TestResult]:
        """
        Run test cases on generated code.
        
        Args:
            code: Generated code
            test_cases: List of test cases
            setup_code: Setup code
            
        Returns:
            List of test results
        """
        results = []
        
        for test in test_cases:
            result = self._run_single_test(code, test, setup_code)
            results.append(result)
        
        return results
    
    def _run_single_test(
        self, 
        code: str, 
        test: str, 
        setup_code: str = ""
    ) -> TestResult:
        """
        Run a single test case.
        
        Args:
            code: Generated code
            test: Test case
            setup_code: Setup code
            
        Returns:
            Test result
        """
        start_time = time.time()
        
        try:
            # Create execution namespace
            namespace = {}
            
            # Execute setup code
            if setup_code:
                exec(setup_code, namespace)
            
            # Execute generated code
            exec(code, namespace)
            
            # Execute test
            exec(test, namespace)
            
            execution_time = time.time() - start_time
            return TestResult(True, "", execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = self._format_error(e)
            return TestResult(False, error_message, execution_time)
    
    def _format_error(self, error: Exception) -> str:
        """
        Format error message.
        
        Args:
            error: Exception object
            
        Returns:
            Formatted error message
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Truncate long error messages
        if len(error_message) > 200:
            error_message = error_message[:200] + "..."
        
        return f"{error_type}: {error_message}"
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax.
        
        Args:
            code: Python code
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)


class MetricsCalculator:
    """
    Metrics calculation utilities.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_success_rate(self, test_results: List[TestResult]) -> float:
        """
        Calculate success rate.
        
        Args:
            test_results: List of test results
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        if not test_results:
            return 0.0
        
        passed_count = sum(1 for result in test_results if result.passed)
        return passed_count / len(test_results)
    
    def calculate_average_execution_time(self, test_results: List[TestResult]) -> float:
        """
        Calculate average execution time.
        
        Args:
            test_results: List of test results
            
        Returns:
            Average execution time in seconds
        """
        if not test_results:
            return 0.0
        
        total_time = sum(result.execution_time for result in test_results)
        return total_time / len(test_results)
    
    def calculate_error_rate(self, test_results: List[TestResult]) -> float:
        """
        Calculate error rate.
        
        Args:
            test_results: List of test results
            
        Returns:
            Error rate (0.0 to 1.0)
        """
        if not test_results:
            return 0.0
        
        error_count = sum(1 for result in test_results if not result.passed)
        return error_count / len(test_results)
    
    def get_error_summary(self, test_results: List[TestResult]) -> Dict[str, int]:
        """
        Get error type summary.
        
        Args:
            test_results: List of test results
            
        Returns:
            Dictionary of error types and counts
        """
        error_summary = {}
        
        for result in test_results:
            if not result.passed and result.error_message:
                error_type = result.error_message.split(':')[0]
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        return error_summary
    
    def calculate_comprehensive_metrics(
        self, 
        test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics.
        
        Args:
            test_results: List of test results
            
        Returns:
            Dictionary of metrics
        """
        return {
            "success_rate": self.calculate_success_rate(test_results),
            "error_rate": self.calculate_error_rate(test_results),
            "average_execution_time": self.calculate_average_execution_time(test_results),
            "total_tests": len(test_results),
            "passed_tests": sum(1 for result in test_results if result.passed),
            "failed_tests": sum(1 for result in test_results if not result.passed),
            "error_summary": self.get_error_summary(test_results)
        }
