
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Pilih model (bisa diganti dengan model LLM lain)
model_name = "gpt2"  # atau "tiiuae/falcon-rw-1b", "mistralai/Mistral-7B-Instruct-v0.1", dll jika punya GPU besar
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Pastikan model berada di GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt untuk eksplorasi
prompt = "Once upon a time in a distant galaxy"

# Kombinasi parameter untuk eksplorasi
temperatures = [0.7, 1.0, 1.3]
top_ks = [0, 50, 100]
top_ps = [0.8, 0.9, 1.0]
max_length = 50

# Tokenisasi
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Eksplorasi kombinasi parameter
for temp in temperatures:
    for top_k in top_ks:
        for top_p in top_ps:
            print(f"\n=== Temperature: {temp}, Top-k: {top_k}, Top-p: {top_p} ===")
            output = model.generate(
                **inputs,
                do_sample=True,
                max_length=max_length,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id  # untuk menghindari warning
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            print(decoded)
