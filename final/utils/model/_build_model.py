import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ._lora_targets import lora_targets


def build_model(model_name: str, tokenizer):
    """Build model
    
    [Params]
    model_name : str
    tokenizer
    
    [Return]
    model
    """
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=lora_targets(model_name),
        bias='none', task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)
    
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.enable_input_require_grads()
    
    return model
