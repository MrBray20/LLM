# # # can translate german, inggris, 
# # from transformers import T5Tokenizer, T5ForConditionalGeneration

# # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
# # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

# # input_text = "Sentimen :Saya sangat menyukai film ini, ceritanya sangat menarik!"
# # input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# # outputs = model.generate(input_ids)
# # print(tokenizer.decode(outputs[0]))


# from transformers import T5ForConditionalGeneration, T5Tokenizer

# # Memuat model dan tokenizer FLAN-T5-Large
# model_name = "google/flan-t5-large"
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = T5Tokenizer.from_pretrained(model_name)

# # Fungsi untuk melakukan sentimen analisis
# def analyze_sentiment(text):
#     # Format input untuk FLAN-T5
#     input_text = f"Apakah sentimen dari teks berikut: '{text}'? Jawab dengan 'positif' atau 'negatif'."
    
#     # Tokenisasi input
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
#     # Generate output
#     outputs = model.generate(inputs["input_ids"], max_length=10)  # Output berupa label "positif" atau "negatif"
    
#     # Decode output
#     sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return sentiment

# # Contoh teks untuk dianalisis
# text = "Film ini jelek sekali"
# sentiment = analyze_sentiment(text)
# print(f"Sentimen: {sentiment}")


# from transformers import T5ForConditionalGeneration, T5Tokenizer

# # Memuat model dan tokenizer FLAN-T5-Large
# model_name = "google/flan-t5-large"
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = T5Tokenizer.from_pretrained(model_name)

# # Fungsi untuk melakukan sentimen analisis
# def analyze_sentiment(text):
#     # Format input untuk FLAN-T5
#     input_text = f"Tentukan sentimen dari teks berikut: '{text}'. Jawab dengan 'positif' atau 'negatif'."
    
#     # Tokenisasi input
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
#     # Generate output dengan sampling
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=20,  # Tingkatkan max_length
#         do_sample=True,  # Aktifkan sampling
#         temperature=0.7,  # Kontrol kreativitas output
#     )
    
#     # Decode output
#     sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return sentiment

# # Contoh teks untuk dianalisis
# text = "Film ini jelek sekali"
# sentiment = analyze_sentiment(text)
# print(f"Sentimen: {sentiment}")


# from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification,AutoModelForCausalLM, pipeline

# # Memuat model dan tokenizer FLAN-T5-Large
# model_name = "google/flan-t5-large"
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = T5Tokenizer.from_pretrained(model_name)

# Fungsi untuk melakukan sentimen analisis
# def analyze_sentiment(text):
#     # Format input untuk FLAN-T5
#     input_text = f"Determine the sentiment of the following text: '{text}'. Answer with 'positive' or 'negative'."
    
#     # Tokenisasi input
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
#     # Generate output dengan sampling
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=20,  # Tingkatkan max_length
#         do_sample=True,  # Aktifkan sampling
#         temperature=0.7,  # Kontrol kreativitas output
#     )
    
#     # Decode output
#     sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return sentiment

# Contoh teks berbahasa Inggris untuk dianalisis
# texts = [
#     "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
#     "This product is terrible. It broke after just one use and the customer service was unhelpful.",
#     "The food at this restaurant was amazing, but the service was a bit slow.",
#     "I'm really disappointed with this purchase. The quality is much lower than I expected.",
#     "The conference was well-organized, and the speakers were very knowledgeable."
# ]

# # Analisis sentimen untuk setiap teks
# for text in texts:
#     sentiment = analyze_sentiment(text)
#     print(f"Text: {text}")
    # print(f"Sentiment: {sentiment}\n")


# config = AutoConfig.from_pretrained(model_name)

# print(model)
# print(f"Model Name: {config.model_type}")
# print(f"Model Architecture: {model.__class__.__name__}")
# print(f"Model Config: {config}")

# Cetak detail layer dan head
# for name, module in model.named_modules():
#     print(f"Layer: {name}")
#     print(f"  Type: {type(module)}")
#     print(f"  Parameters: {sum(p.numel() for p in module.parameters())}")
#     print(f"  Children: {list(module.children())}")
#     print("\n")
    
# print(model.forward)
# model_name = "google/flan-t5-large"
# pipe = pipeline(task="sentiment-analysis",model=model_name)

# texts = [
#     "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
#     "This product is terrible. It broke after just one use and the customer service was unhelpful.",
#     "The food at this restaurant was amazing, but the service was a bit slow.",
#     "I'm really disappointed with this purchase. The quality is much lower than I expected.",
#     "The conference was well-organized, and the speakers were very knowledgeable."
# ]


# result= pipe(texts)

# print(result)



from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModel 
from torchinfo import summary


model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


text = "Write about cats in 100 words."

# tokens = tokenizer.tokenize(kalimat)
# token_ids = tokenizer.encode(kalimat)

summary(model)

# print("Tokens:",tokens)
# print("token IDs:", token_ids)

# input = tokenizer(kalimat,return_tensors='pt')

# print("Input dict:",input)

# print()


# token_embeddings = model.shared(torch.tensor(token_ids)).unsqueeze(0)

# print("Token embeddings shape:", token_embeddings.shape)

# position_ids = torch.arange(len(token_ids)).unsqueeze(0)
# position_embeddings = model.encoder.embed_positions(position_ids)

# 2. Tokenisasi
# inputs = tokenizer(text, return_tensors="pt")
# print("Tokens:", tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
# print("Token IDs:", inputs.input_ids)

# # 3. Dapatkan Token Embedding (tanpa positional)
# token_embeddings = model.shared(inputs.input_ids)
# print("Token embeddings shape:", token_embeddings.shape)  # [1, seq_len, d_model]

# # 4. Proses Encoder (termasuk positional embedding di dalamnya)
# with torch.no_grad():
#     encoder_outputs = model.encoder(
#         input_ids=inputs.input_ids,
#         attention_mask=inputs.attention_mask
#     )

# # Akses last_hidden_state
# last_hidden = encoder_outputs.last_hidden_state
# print("Shape last_hidden_state:", last_hidden.shape)  # Contoh: [1, 8, 1024]

# # Lihat 5 nilai pertama dari token pertama
# print("Sample values (token 0, 5 first dims):", last_hidden[0, 0, :5])
