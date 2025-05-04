# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # import transformers
# # import torch

# # model = "tiiuae/falcon-7b"

# # tokenizer = AutoTokenizer.from_pretrained(model)
# # pipeline = transformers.pipeline(
# #     "text-generation",
# #     model=model,
# #     tokenizer=tokenizer,
# #     torch_dtype=torch.bfloat16,
# #     trust_remote_code=True,
# #     device_map="auto",
# # )
# # sequences = pipeline(
# #    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
# #     max_length=200,
# #     do_sample=True,
# #     top_k=10,
# #     num_return_sequences=1,
# #     eos_token_id=tokenizer.eos_token_id,
# # )
# # for seq in sequences:
# #     print(f"Result: {seq['generated_text']}")

# # import time
# # print("Loading model...")
# # start_time = time.time()

# # from transformers import pipeline

# # pipe = pipeline("text-generation", model="tiiuae/falcon-7b", device=0)

# # print("Model loaded in", round(time.time() - start_time, 2), "seconds.")

# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# model_name = "tiiuae/falcon-7b"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     load_in_8bit=True,  # Bisa juga load_in_4bit=True
#     device_map="auto"
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# prompt = "Tell me about AI:"
# result = pipe(prompt, max_length=100, truncation=True)

# print("Generated Text:\n", result[0]['generated_text'])

# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# model_name = "tiiuae/falcon-7b"

# # Konfigurasi quantization agar lebih hemat memori
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Gunakan 4-bit quantization
#     llm_int8_enable_fp32_cpu_offload=True  # Aktifkan CPU offloading
# )

# # Load model dengan CPU offloading
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto"  # Secara otomatis bagi beban antara CPU & GPU
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# prompt = "Tell me about AI:"
# result = pipe(prompt, max_length=100, truncation=True)

# print("Generated Text:\n", result[0]['generated_text'])

# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# import torch

# model_name = "tiiuae/falcon-7b"

# # Pastikan dtype sesuai agar tidak error
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,  # Pastikan menggunakan float16
#     llm_int8_enable_fp32_cpu_offload=True
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto"
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# prompt = "Tell me about AI:"
# result = pipe(prompt, max_length=100, truncation=True)

# print("Generated Text:\n", result[0]['generated_text'])

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# input_text = "write a storytelling"
# input_ids = tokenizer(input_text, return_tensors="pt")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))


# pip install accelerate
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))

# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a teacher",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...

