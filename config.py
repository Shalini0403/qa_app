# config.py
import json
import os
from typing import Dict


def load_config(config_path: str = "config.json") -> Dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        return json.load(f)