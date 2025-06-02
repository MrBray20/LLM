from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Format chat-style prompt
instruction = "Jelaskan secara sederhana apa itu machine learning."
prompt = f"<|system|>\nKamu adalah asisten AI yang membantu pengguna.\n<|user|>\n{instruction}\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
