
import torch
from transformers import pipeline

# Load model dan tokenizer
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Format input chat
messages = [
    {"role": "system", "content": "You are a very good writer. Write a story about Tarzan of the Jungle!"},
    {"role": "user", "content": "Describe Tarzan's adventures in the jungle, his encounters with wild animals, and his friendship with Jane."},
]

# Generate teks lanjutan
outputs = pipe(
    messages,
    max_new_tokens=500,  # Batas maksimal yang cukup besar
    do_sample=True,      # Aktifkan sampling
    top_k=50,            # Batasi sampling ke 50 token teratas
    top_p=0.95,          # Gunakan nucleus sampling dengan p=0.95
    temperature=0.9,     # Kontrol kreativitas output
    eos_token_id=pipe.tokenizer.eos_token_id,  # Gunakan eos_token_id untuk menghentikan generasi
    early_stopping=True  # Berhenti saat menemukan eos_token_id
)

# Tampilkan hasil
print("\nGenerated story:")
print(outputs[0]["generated_text"][-1]["content"])