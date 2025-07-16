
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
modelMistral=AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
tokenMistral=AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")

modelLLaMA=AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
tokekenLLaMA=AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct-bnb-4bit")

modelGemma=AutoModelForCausalLM.from_pretrained("unsloth/gemma-2b-it-bnb-4bit")
tokenGemma=AutoTokenizer.from_pretrained("unsloth/gemma-2b-it-bnb-4bit")


pipelineMistral= pipeline("text-generation", model=modelMistral, tokenizer=tokenMistral, device_map="auto",torch_dtype=torch.float16)

pipelineLLaMA= pipeline("text-generation", model=modelLLaMA, tokenizer=tokekenLLaMA, device_map="auto",torch_dtype=torch.float16)

pipelineGemma= pipeline("text-generation", model=modelGemma, tokenizer=tokenGemma, device_map="auto",torch_dtype=torch.float16)


prompt = f"""
Analyze the sentiment of the following text and classify it as [POSITIVE/NEGATIVE/NEUTRAL]. 
Provide the answer in JSON format with keys: "sentiment", "confidence" (0-1), and "explanation".

Text: "I absolutely loved this movie! The acting was fantastic and the story was captivating."

Answer:"""

outputmistral = pipelineMistral(prompt,max_new_tokens=200,temperature=0.7,do_sample=True)
outputLLaMA = pipelineLLaMA(prompt,max_new_tokens=200,temperature=0.7,do_sample=True)
outputGemma = pipelineGemma(prompt,max_new_tokens=200,temperature=0.7,do_sample=True)

print(">>>>>>>>>>>>>>>>>>>>>Mistral>>>>>>>>>>>>>>>>>>>")
print(outputmistral[0]['generated_text'])
print(">>>>>>>>>>>>>>>>>>>>>LLaMA>>>>>>>>>>>>>>>>>>>")
print(outputLLaMA[0]['generated_text'])
print(">>>>>>>>>>>>>>>>>>>>>Gemma>>>>>>>>>>>>>>>>>>>")
print(outputGemma[0]['generated_text'])