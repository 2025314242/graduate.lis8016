from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils.config import load_config


def load_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Load tokenizer
    
    [Param]
    model_name : str
    
    [Return]
    tokenizer : PreTrainedTokenizerFast
    """
    config = load_config()
    rationale_labels = config['SPECIAL_TOKENS']['RATIONALE']
    output_labels = config['SPECIAL_TOKENS']['OUTPUT']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({
        'additional_special_tokens': rationale_labels + output_labels
    })
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer
