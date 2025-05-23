# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", trust_remote_code=True)
pipe(messages)