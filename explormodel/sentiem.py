from transformers import pipeline

# Muat model yang telah dilatih untuk klasifikasi sentimen
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)

# Teks yang akan dianalisis
texts = [
    "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
    "This product is terrible. It broke after just one use and the customer service was unhelpful.",
    "The food at this restaurant was amazing, but the service was a bit slow.",
    "I'm really disappointed with this purchase. The quality is much lower than I expected.",
    "The conference was well-organized, and the speakers were very knowledgeable."
]

# Lakukan sentimen analisis
result = classifier(texts)

# Cetak hasil
print(result)