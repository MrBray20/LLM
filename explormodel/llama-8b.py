# # # # Use a pipeline as a high-level helper
# from transformers import pipeline

# messages = """Translate the English text to French.
# Text: Sometimes, I've believed as many as six impossible things before breakfast.
# Translation:
# """
# pipe = pipeline(task=, model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")


# print(pipe(messages))



# # from transformers import LlamaForQuestionAnswering

# # # Muat model yang sudah difine-tune (atau model dasar)
# # model = LlamaForQuestionAnswering.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

# # # Cetak semua lapisan
# # print(model)

# # from transformers import pipeline

# # # Gunakan base model tanpa fine-tuning
# # classifier = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")

# # # Analisis sentimen via prompt engineering
# # prompt = """
# # Tentukan sentimen kalimat berikut (positif/negatif/netral):
# # "Pengiriman sangat lambat dan tidak profesional."
# # Jawaban: 
# # """

# # result = classifier(prompt, max_length=50)
# # print(result[0]['generated_text'])

# # Use a pipeline as a high-level helper
# # from transformers import pipeline

# # pipe = pipeline("text-generation", model="unsloth/Meta-Llama-3.1-8B")

# # print(pipe)