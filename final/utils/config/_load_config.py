import yaml
from typing import Any, Dict

def load_config(path: str='.config.yaml') -> Dict[str, Any]:
    """Load config
    
    [Param]
    path : str = '.config.yaml'
    
    [Return]
    config : Dict[str, Any]
    """
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config
