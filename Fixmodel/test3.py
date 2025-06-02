from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-small"

tokenizer_manual = AutoTokenizer.from_pretrained(model_name)

model_manual = AutoModelForSeq2SeqLM.from_pretrained(model_name)

instruction = "Terjemahkan teks berikut ke dalam bahasa Inggris:"
text_to_process = "Indonesia adalah negara kepulauan yang indah."

full_input_text = f"{instruction} {text_to_process}"

inputs_manual = tokenizer_manual(full_input_text, return_tensors="pt")

with torch.no_grad():
    output_tokens = model_manual.generate(
        inputs_manual["input_ids"],
        max_new_tokens=50,  
        num_beams=5,        
        early_stopping=True 
    )

generated_text_manual = tokenizer_manual.decode(output_tokens[0], skip_special_tokens=True)

print(f"Instruksi dan Teks Input: {full_input_text}")
print(f"Hasil Generasi Model: {generated_text_manual}")