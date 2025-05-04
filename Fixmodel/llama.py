from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class LLAMA:
    def __init__(self,model_name="meta-llama/Llama-3.2-3B-Instruct"):
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        
    def generateText(self,prompt):
        input_ids = self.tokenizer(prompt,return_tensors="pt").to("cuda")
        
        output = self.model.generate(
            input_ids=input_ids.input_ids,
            max_new_tokens=100,
            temperature=0.7,     
            do_sample=False,
            early_stopping=True,  # Berhenti jika model menganggap sudah selesai
            eos_token_id=self.tokenizer.eos_token_id,  # Token akhir generasi
            pad_token_id=self.tokenizer.eos_token_id       
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)