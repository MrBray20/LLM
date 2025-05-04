# Use a pipeline as a high-level helper
# from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer
# from torchinfo import summary
# # messages = """ You are a very good writer. Write a story about Tarzan of the Jungle.
# #     Describe Tarzan's adventures in the jungle, his encounters with wild animals, and his friendship with Jane.
# # """
# # pipe = pipeline("text-generation", model="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")

# # print(pipe(messages))


# # model = AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
# # tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
# # print(model.forward)
# # summary(model)

# # def analyze_sentiment(text):
# #     # Format input untuk FLAN-T5
# #     input_text = f"Determine the sentiment of the following text: '{text}'. Answer with 'positive' or 'negative'."
    
# #     # Tokenisasi input
# #     inputs = tokenizer(input_text, return_tensors="pt")
    
# #     # Generate output dengan sampling
# #     outputs = model.generate(
# #         inputs["input_ids"],
# #         do_sample=True,  # Aktifkan sampling
# #         temperature=0.7,  # Kontrol kreativitas output
# #     )
    
# #     # Decode output
# #     sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #     return sentiment


# # model = AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
# # tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
# # # pipe = pipeline(task="sentiment-analysis", model="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")

# # texts = [
# #     "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
# #     "This product is terrible. It broke after just one use and the customer service was unhelpful.",
# #     "The food at this restaurant was amazing, but the service was a bit slow.",
# #     "I'm really disappointed with this purchase. The quality is much lower than I expected.",
# #     "The conference was well-organized, and the speakers were very knowledgeable."
# # ]

# # for text in texts:
# #     sentiment = analyze_sentiment(text)
# #     print(f"Text: {text}")
# #     print(f"Sentiment: {sentiment}\n")




# def analyze_sentiment():
#     # Format input untuk FLAN-T5
#     prompt = f"""
#         Analyze the sentiment of the following text and classify it as [POSITIVE/NEGATIVE/NEUTRAL]. 
#         Provide the answer in JSON format with keys: "sentiment", "confidence" (0-1), and "explanation".

#         Text: "I absolutely loved this movie! The acting was fantastic and the story was captivating."

#         Answer:
#         """
#     # Tokenisasi input
#     inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    
#     # Move input tensors to the same device as the model
#     # inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
#     # Generate output dengan sampling
#     outputs = model.generate(
#         inputs["input_ids"],
#         do_sample=True,  # Aktifkan sampling
#         temperature=0.7,  # Kontrol kreativitas output
#     )
    
#     # Decode output
#     sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return sentiment

# model = AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
# tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")

# # Ensure the model is on the GPU


# texts = [
#     "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
#     "This product is terrible. It broke after just one use and the customer service was unhelpful.",
#     "The food at this restaurant was amazing, but the service was a bit slow.",
#     "I'm really disappointed with this purchase. The quality is much lower than I expected.",
#     "The conference was well-organized, and the speakers were very knowledgeable."
# ]

# print(analyze_sentiment())