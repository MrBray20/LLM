# # from transformers import AutoTokenizer, AutoModelForCausalLM

# # tokenizer = AutoTokenizer.from_pretrained("eunyounglee/GPT-NeoX-1.3B-2GB-Eng")
# # model = AutoModelForCausalLM.from_pretrained("eunyounglee/GPT-NeoX-1.3B-2GB-Eng")

# # # Tokenisasi input
# # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# # # Hapus token_type_ids jika ada
# # inputs.pop("token_type_ids", None)

# # # Generate teks
# # generated_ids = model.generate(**inputs, max_length=50)

# # # Decode hasil output
# # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# # # Tampilkan hasil
# # print(generated_text)



# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load model & tokenizer
# tokenizer = AutoTokenizer.from_pretrained("eunyounglee/GPT-NeoX-1.3B-2GB-Eng")
# model = AutoModelForCausalLM.from_pretrained("eunyounglee/GPT-NeoX-1.3B-2GB-Eng")

# # Prompt input (bisa kamu ubah sesuai kebutuhan)
# prompt = "can u tell me story about herry potter"

# # Tokenisasi
# inputs = tokenizer(prompt, return_tensors="pt")

# # Hapus token_type_ids karena tidak didukung oleh GPT-NeoX
# inputs.pop("token_type_ids", None)

# # Generate output seperti ChatGPT-style
# generated_ids = model.generate(
#     **inputs,
#     eos_token_id=tokenizer.eos_token_id,          # biarkan model tahu kapan berhenti
#     pad_token_id=tokenizer.eos_token_id,          # untuk mencegah warning dari generate()
#     max_new_tokens=200,                           # panjang maksimal, tapi model bisa berhenti lebih cepat
#     do_sample=True,                               # sampling agar hasilnya lebih variatif
#     temperature=0.7,                              # kreativitas jawaban
#     top_p=0.9,                                    # nucleus sampling
#     early_stopping=True                           # berhenti kalau output sudah "cukup"
# )

# # Decode ke bentuk teks
# generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# # Print hasilnya
# print(generated_text)



from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("eunyounglee/GPT-NeoX-1.3B-2GB-Eng")
model = AutoModelForCausalLM.from_pretrained("eunyounglee/GPT-NeoX-1.3B-2GB-Eng")

# Ubah prompt sesuai keinginan
# prompt = "Tell me a story about Harry Potter"
prompt = "Once upon a time, there was a young wizard named Harry Potter who lived in a magical world. One day,"


# Tokenisasi prompt
inputs = tokenizer(prompt, return_tensors="pt")
inputs.pop("token_type_ids", None)

# Generate output
generated_ids = model.generate(
    **inputs,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    early_stopping=True
)

# Decode output
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
