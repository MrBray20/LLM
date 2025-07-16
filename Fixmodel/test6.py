
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

prompt = f"""Instruction:
Create a story about time travel with the following elements:

1. A character finds a way to travel to the past.
2. They try to change an important event in their life.
3. The change has unintended consequences in the present.
4. The story ends with a realization or emotional resolution.

Answer:"""

outputmistral = pipelineMistral(prompt,max_new_tokens=512, temperature=0.7,top_p=0.95,top_k=50,repetition_penalty=1.1,do_sample=True, eos_token_id=tokenMistral.eos_token_id)
outputLLaMA = pipelineLLaMA(prompt,max_new_tokens=512, temperature=0.7,top_p=0.95,top_k=50,repetition_penalty=1.1,do_sample=True, eos_token_id=tokekenLLaMA.eos_token_id)
outputGemma = pipelineGemma(prompt,max_new_tokens=512, temperature=0.7,top_p=0.95,top_k=50,repetition_penalty=1.1,do_sample=True, eos_token_id=tokenGemma.eos_token_id)

print(">>>>>>>>>>>>>>>>>>>>>Mistral>>>>>>>>>>>>>>>>>>>")
print(outputmistral[0]['generated_text'])
print(">>>>>>>>>>>>>>>>>>>>>LLaMA>>>>>>>>>>>>>>>>>>>")
print(outputLLaMA[0]['generated_text'])
print(">>>>>>>>>>>>>>>>>>>>>Gemma>>>>>>>>>>>>>>>>>>>")
print(outputGemma[0]['generated_text'])