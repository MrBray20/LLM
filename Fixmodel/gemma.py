from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

class Gemma:
    def __init__(self, model_name="google/gemma-2b-it"):
        
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        self.modelname=model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)
        self.model.to(self.device)
        
        
    def generateText(self,prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            input_ids=input_ids.input_ids
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generateTextPipe(self,prompt):
        # input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        generator = pipeline(
            "text-generation",
            model=self.modelname,
            device_map="auto",  # Gunakan GPU jika tersedia
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16
        )

        # Generate teks
        result = generator(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
        
        return (result[0]['generated_text'])