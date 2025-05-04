# # # # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# # # tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
# # # model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")

# # # Load model directly

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit")
# model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit")  


# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


# # from transformers import AutoTokenizer, AutoModelForCausalLM

# # tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
# # model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

# def analyze_sentiment(text):
#     # Format input untuk FLAN-T5
#     input_text = f"Classify the sentiment of this text as 'positive' or 'negative' only: '{text}'"
#     # Tokenisasi input
#     inputs = tokenizer(input_text, return_tensors="pt")
    
#     # Move input tensors to the same device as the model
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
#     # Generate output dengan sampling
#     outputs = model.generate(
#         inputs["input_ids"],
#         do_sample=True,  # Aktifkan sampling
#         temperature=0.7,  # Kontrol kreativitas output
#     )
    
#     # Decode output
#     sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return sentiment

# # Ensure the model is on the GPU
# # model.to('cuda')

# texts = [
#     "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
#     "This product is terrible. It broke after just one use and the customer service was unhelpful.",
#     "The food at this restaurant was amazing, but the service was a bit slow.",
#     "I'm really disappointed with this purchase. The quality is much lower than I expected.",
#     "The conference was well-organized, and the speakers were very knowledgeable."
# ]

# for text in texts:
#     sentiment = analyze_sentiment(text)
#     print(f"Text: {text}")
#     print(f"Sentiment: {sentiment}\n")



# # from transformers import AutoTokenizer, AutoModelForCausalLM

# # tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
# # model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
# #     device_map="auto",  # Otomatis bagi modul ke GPU/CPU
# #     load_in_4bit=True
# # )
# # def analyze_sentiment(text):
# #     # Format prompt sesuai instruksi Llama-3
# #     messages = [
# #         {"role": "system", "content": "You are a sentiment analysis assistant. Classify the sentiment as 'positive' or 'negative'."},
# #         {"role": "user", "content": f"Analyze the sentiment of this text: '{text}'"}
# #     ]
    
# #     # Tokenisasi dengan format chat
# #     input_text = tokenizer(messages, tokenize=False)
# #     inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
# #     # Generate output
# #     outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.7)
    
# #     # Decode dan ekstrak jawaban
# #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #     return response.split("assistant\n")[-1].strip()  # Ambil bagian setelah "assistant"

# # texts = [
# #     "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
# #     "This product is terrible. It broke after just one use and the customer service was unhelpful.",
# #     "The food at this restaurant was amazing, but the service was a bit slow.",
# #     "I'm really disappointed with this purchase. The quality is much lower than I expected.",
# #     "The conference was well-organized, and the speakers were very knowledgeable."
# # ]

# # for text in texts:
# #     sentiment = analyze_sentiment(text)
# # #     print(f"Text: {text}")
# # #     print(f"Sentiment: {sentiment}\n")


# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16"
# )
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-3B-Instruct",
#     quantization_config=bnb_config,  # Aktifkan 4-bit
#     device_map="auto"              # Auto-load ke GPU/CPU
# )

# # Set prompt untuk inference
# prompt = f"""
# Analyze the sentiment of the following text and classify it as [POSITIVE/NEGATIVE/NEUTRAL]. 
# Provide the answer in JSON format with keys: "sentiment", "confidence" (0-1), and "explanation".

# Text: "I'm really disappointed with this purchase. The quality is much lower than I expected."

# Answer:
# """

# input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")  # Pindah ke GPU

# # Generate respons
# output = model.generate(
#     input_ids=input_ids.input_ids,
#     max_new_tokens=2000,  # Batas token output
#     temperature=0.7,     # Kontrol kreativitas (0.1-1.0)
#     do_sample=False,       # Aktifkan sampling untuk hasil lebih alami
#     early_stopping=True,  # Berhenti jika model menganggap sudah selesai
#     eos_token_id=tokenizer.eos_token_id,  # Token akhir generasi
#     pad_token_id=tokenizer.eos_token_id
# )

# # Decode dan cetak hasil
# response = tokenizer.decode(output[0], skip_special_tokens=True)
# print(response)


from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
import torch

# Konfigurasi 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer dan model
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Buat pipeline untuk text-generation
pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    device_map="auto",  # Auto-load ke GPU jika tersedia
    # quantization_config=bnb_config,  # Aktifkan 4-bit
    torch_dtype=torch.float16
)

# Contoh penggunaan
prompt = """Analyze the sentiment of the following text and classify it as [POSITIVE/NEGATIVE/NEUTRAL]. 
Provide the answer in JSON format with keys: "sentiment", "confidence" (0-1), and "explanation".

Text: "I'm really disappointed with this purchase. The quality is much lower than I expected."

Answer:"""

output = pipe(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

print(output[0]['generated_text'])
