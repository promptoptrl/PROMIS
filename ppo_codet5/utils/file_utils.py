"""
File utility functions.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: File path
        indent: JSON indentation
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: File path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: File path
    """
    import pickle
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: File path
        
    Returns:
        Loaded data
    """
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)
