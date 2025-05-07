from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 1. Pilih model yang mendukung causal LM dan juga bisa dipakai sebagai AutoModel
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"  # Bisa diganti dengan LLaMA atau Mistral jika tersedia lokal
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
    
)

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Contoh input
input_text = "The capital of France is"
inputs = tokenizer(input_text, return_tensors="pt")

# -----------------------------------
# 🔹 A. Menggunakan AutoModel (hanya hidden states)
# -----------------------------------
model_encoder = AutoModel.from_pretrained(model_name).to("cuda")
inputs = {key: value.to("cuda") for key, value in inputs.items()}
with torch.no_grad():
    outputs_encoder = model_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

print(model_encoder)
print("AutoModel Output (Hidden State Shape):", outputs_encoder.last_hidden_state.shape)

# -----------------------------------
# 🔹 B. Menggunakan AutoModelForCausalLM (dapat generate teks)
# -----------------------------------
model_lm = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
inputs = {key: value.to("cuda") for key, value in inputs.items()}
with torch.no_grad():
    outputs_lm = model_lm(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

print(model_lm)
print("AutoModelForCausalLM Output (Logits Shape):", outputs_lm.logits.shape)

# -----------------------------------
# 🔹 Generate Text dengan CausalLM
# -----------------------------------
generated_ids = model_lm.generate(**inputs, max_new_tokens=20)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("\nGenerated Text:")
print(generated_text)
