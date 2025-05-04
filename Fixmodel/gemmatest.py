
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-9b-it-bnb-4bit")
# model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-2-9b-it-bnb-4bit")

# def analyze_sentiment(text):
#     # Format input untuk FLAN-T5
#     input_text = f"Determine the sentiment of the following text: '{text}'. Answer with 'positive' or 'negative'."
    
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
# model.to('cuda')

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



# # class Gemma:
# #     def __init__(self),model_name=:


# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch


# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it-pytorch")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it-pytorch")

# Load model dan tokenizer (pastikan GPU tersedia)
# tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-9b-it-bnb-4bit")
# model = AutoModelForCausalLM.from_pretrained(
#     "unsloth/gemma-2-9b-it-bnb-4bit",
#     device_map="auto",  # Otomatis ke GPU jika tersedia
#     torch_dtype=torch.float16,
# )

# def analyze_sentiment():
#     # Format instruksi untuk Gemma (chat model)
#     prompt = f"""
#         Analyze the sentiment of the following text and classify it as [POSITIVE/NEGATIVE/NEUTRAL]. 
#         Provide the answer in JSON format with keys: "sentiment", "confidence" (0-1), and "explanation".

#         Text: "I absolutely loved this movie! The acting was fantastic and the story was captivating."

#         Answer:
#         """
    
#     # Tokenisasi
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
#     # Generate output
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=10,  # Batasi panjang output
#         do_sample=True,
#         temperature=0.7,
#     )
#     sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return sentiment

    # Decode dan ekstrak jawaban
    # full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # sentiment = full_response.split("model")[-1].strip().lower()  # Ambil bagian setelah "model"
    
    # # Pastikan hanya "positive" atau "negative"
    # if "positive" in sentiment:
    #     return "positive"
    # elif "negative" in sentiment:
    #     return "negative"
    # else:
    #     return "neutral"  # Fallback

# Contoh teks
# texts = [
#     "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
#     "This product is terrible. It broke after just one use and the customer service was unhelpful.",
#     "The food at this restaurant was amazing, but the service was a bit slow.",
#     "I'm really disappointed with this purchase. The quality is much lower than I expected.",
#     "The conference was well-organized, and the speakers were very knowledgeable."
# ]

# # Analisis
# for text in texts:
#     sentiment = analyze_sentiment(text)
#     print(f"Text: {text[:100]}...")  # Potong teks panjang
    # print(f"Sentiment: {sentiment}\n")
    
# print(analyze_sentiment())

# import torch
# from transformers import pipeline

# pipe = pipeline(
#     "text-generation",
#     model="google/gemma-2-2b-it",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",  # replace with "mps" to run on a Mac device
# )

# messages = [
#     {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
# ]

# outputs = pipe(messages, max_new_tokens=256)
# assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
# print(assistant_response)
# # Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Disable Dynamo errors (Solution 2)
# torch._dynamo.config.suppress_errors = True

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto"  # Automatically handles CPU/GPU
)

def analyze_sentiment():
    prompt = f"""
        Analyze the sentiment of the following text and classify it as [POSITIVE/NEGATIVE/NEUTRAL]. 
        Provide the answer in JSON format with keys: "sentiment", "confidence" (0-1), and "explanation".

        Text: "I absolutely loved this movie! The acting was fantastic and the story was captivating."

        Answer:
        """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():  # Reduces VRAM usage
        outputs = model.generate(**inputs, max_new_tokens=100)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(analyze_sentiment())
