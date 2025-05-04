from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
help(generator)