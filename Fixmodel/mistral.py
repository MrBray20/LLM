from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Mistral:
    def __init__(self,model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        
    def generateText(self,prompt):
        input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            input["input_ids"],
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(output[0],skip_special_tokens=True)