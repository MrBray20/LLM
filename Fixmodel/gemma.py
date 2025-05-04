from transformers import BitsAndBytesConfig
from ModelStruktur import ModelStruktur


class Gemma(ModelStruktur):
    def __init__(self,model_name="google/gemma-2b-it"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        model_kwargs = {
            "quantization_config":bnb_config
        }
        
        super().__init__(model_name,model_kwargs)

        
    