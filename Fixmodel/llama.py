from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from ModelStruktur import ModelStruktur

class LLAMA(ModelStruktur):
    def __init__(self,model_name="meta-llama/Llama-3.2-3B-Instruct"):
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        
        model_kwargs={
            "quantization_config":bnb_config
        }
        
        super().__init__(model_name,model_kwargs)