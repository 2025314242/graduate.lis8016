from typing import List


def lora_targets(model_name: str) -> List[str]:
    """Lora targets
    
    [Param]
    model_name : str
    
    [Return]
    lora_targets : List[str]
    """
    if 'gemma' in model_name:
        return ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj']
    
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']
