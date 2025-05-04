
from transformers import pipeline

# Load pipeline untuk text generation
generator = pipeline("text-generation")

# Prompt awal
prompt = "In the future, artificial intelligence will"

# Generate teks
results = generator(prompt, max_length=500, num_return_sequences=1)

# Tampilkan hasil
print("Generated Text:")
print(results[0]['generated_text'])


# from transformers import pipeline

# # Load pipeline untuk sentiment analysis
# sentiment_analyzer = pipeline("sentiment-analysis")

# # Kalimat yang ingin dianalisis
# texts = [
#     "I love using Hugging Face Transformers!",
#     "This is the worst experience I've ever had."
# ]

# # Analisis sentimen
# results = sentiment_analyzer(texts)

# # Tampilkan hasil
# for text, result in zip(texts, results):
#     print(f"Text: {text}")
#     print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")
