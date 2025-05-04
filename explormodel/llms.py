# # pip install accelerate
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch._dynamo
# from accelerate import dispatch_model



# torch._dynamo.config.suppress_errors = True


# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2-2b-it",
#     device_map="auto",
# )
# model = dispatch_model(model, device="cuda")
# print(next(model.parameters()).device)  # Harusnya cuda:0
# # model.to("cuda")

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids, max_new_tokens=32)

# print(tokenizer.decode(outputs[0]))


# import torch
# print(torch.__version__)  # Cek versi PyTorch
# print(torch.cuda.is_available())  # Cek apakah GPU terdeteksi
# # print(torch.version.cuda)  # Cek versi CUDA yang digunakan PyTorch

# import torch
# print(torch.cuda.memory_summary())

# import torch
# print(torch.cuda.is_available())  # Harusnya True
# print(torch.cuda.device_count())  # Harusnya >= 1
# print(torch.cuda.get_device_name(0))  # Harusnya menunjukkan GPU (misal: RTX 2050)
# print(next(model.parameters()).device)  # Harusnya cuda:0


# from transformers import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)


from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer dan model GPT-2 untuk text generation
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


# Teks input sebagai awal generasi
text = "tell me a story a bout the 'sand king'"

# Encode teks ke tensor PyTorch
input_ids = tokenizer.encode(text, return_tensors='pt')

# Generate teks menggunakan model
# output = model.generate(input_ids, max_length=50, num_return_sequences=1)
output = model.generate(
    input_ids, 
    max_length=1000,  # Bisa diperpanjang jika cerita masih terlalu pendek
    temperature=0.7,  # Menjaga variasi tanpa terlalu acak
    top_k=50,  # Mengambil hanya dari 50 token dengan probabilitas tertinggi
    top_p=0.9,  # Nucleus sampling untuk mengontrol diversitas
    repetition_penalty=1.2,  # Mencegah pengulangan
    do_sample=True  # Mengaktifkan sampling agar output lebih variatif
)


# Decode hasilnya menjadi teks
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
