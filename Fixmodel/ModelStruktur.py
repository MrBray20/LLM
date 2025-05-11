from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class ModelStruktur:
    def __init__(self,model_name, model_kwargs=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_kwargs is None:
            model_kwargs={}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # self.model.to(self.device)
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
    def generateText(self,prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            input_ids=input_ids.input_ids,
            max_new_tokens=100
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generateTextPipe(self,prompt):
        # Generate teks
        result = self.pipeline(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        return (result[0]['generated_text'])
    
    # def generateSentimen(self,prompt):
    #     generator = pipeline(
    #         "text-classification",
    #         model=self.model,
    #         device_map="auto",
    #         tokenizer=self.tokenizer,
    #         torch_dtype=torch.float16
    #     )
        
    #     result = generator(
    #         prompt
            
    #     )
    #     return result