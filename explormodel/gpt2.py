
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
# import tensorflow as tf

# # Load tokenizer dan model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# model = TFGPT2LMHeadModel.from_pretrained('gpt2-xl')

# # Masukkan teks awal sebagai prompt
# prompt = (
#     "You are a very good writer. Write a story about Tarzan of the Jungle. "
#     "Describe Tarzan's adventures in the jungle, his encounters with wild animals, and his friendship with Jane."
# )

# # Tokenisasi input dengan attention mask
# input_ids = tokenizer.encode(prompt, return_tensors='tf', max_length=512, truncation=True)
# attention_mask = tf.ones_like(input_ids)  # Buat attention mask

# # Generate teks lanjutan
# output_ids = model.generate(
#     input_ids,
#     attention_mask=attention_mask,  # Tambahkan attention mask
#     max_length=1000,  # Tetapkan batas maksimal yang cukup besar
#     do_sample=True,
#     top_k=50,
#     top_p=0.95,
#     temperature=0.9,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,  # Gunakan eos_token_id untuk menghentikan generasi
#     early_stopping=True  # Berhenti saat menemukan eos_token_id
# )

# # Decode hasil menjadi teks
# generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# # Tampilkan hasil
# print("\nGenerated story:")
# print(generated_text)


# from transformers import pipeline, set_seed

# # Atur seed agar hasil bisa direproduksi (opsional)
# set_seed(42)

# # Buat pipeline untuk text generation dengan model gpt2-xl
# generator = pipeline("text-generation", model="gpt2-xl")

# # Masukkan teks awal sebagai prompt
# prompt = (
#     "You are a very good writer. Write a story about Tarzan of the Jungle. "
#     "Describe Tarzan's adventures in the jungle, his encounters with wild animals, and his friendship with Jane."
# )

# # Generate teks lanjutan
# results = generator(
#     prompt,
#     max_length=1000,
#     do_sample=True,
#     top_k=50,
#     top_p=0.95,
#     temperature=0.9,
#     num_return_sequences=1,
#     eos_token_id=50256  # GPT2's EOS token
# )

# # Tampilkan hasil
# print("\nGenerated story:")
# print(results[0]['generated_text'])

from transformers import pipeline

# Inisialisasi pipeline untuk text generation dengan model GPT-2 XL
generator = pipeline(
    'text-generation',
    model='gpt2-xl',
    tokenizer='gpt2-xl',
    device=-1  # Gunakan -1 untuk CPU, 0 untuk GPU jika tersedia
)

# Prompt untuk cerita
prompt = (
    "You are a very good writer. Write a story about Tarzan of the Jungle. "
    "Describe Tarzan's adventures in the jungle, his encounters with wild animals, and his friendship with Jane."
)

# Generate teks menggunakan pipeline
generated_texts = generator(
    prompt,
    max_length=1000,  # Panjang maksimal teks yang dihasilkan
    do_sample=True,
    eos_token_id=50256,  # eos_token_id untuk GPT-2
    early_stopping=True
)

# Tampilkan hasil
print("\nGenerated story:")
print(generated_texts[0]['generated_text'])