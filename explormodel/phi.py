
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

# Konfigurasi quantization dengan CPU offloading
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Aktifkan quantization INT8
    llm_int8_enable_fp32_cpu_offload=True  # Aktifkan CPU offloading
)

# Memuat model dengan quantization dan CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=quantization_config,
    device_map="auto"  # Otomatis memetakan model ke GPU/CPU
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Membuat pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt yang lebih spesifik
prompt = (
    "You are a very good writer. Write a story about Tarzan of the Jungle. "
    "Describe Tarzan's adventures in the jungle, his encounters with wild animals, and his friendship with Jane."
)

# Generate teks dengan parameter yang disesuaikan
# result = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.8)
result = pipe(
    prompt,
    max_new_tokens=1000,  # Batas maksimal token yang dihasilkan
    do_sample=True,      # Aktifkan sampling
    temperature=0.9,     # Kontrol kreativitas output
    top_k=50,            # Batasi sampling ke 50 token teratas
    top_p=0.95,          # Gunakan nucleus sampling dengan p=0.95
    eos_token_id=tokenizer.eos_token_id,  # Gunakan eos_token_id untuk menghentikan generasi
    early_stopping=True,  # Berhenti saat menemukan eos_token_id
    num_return_sequences=1
)
# Cetak hasil
print(result[0]["generated_text"])

