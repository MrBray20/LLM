# # Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="unsloth/gemma-2-9b-it-bnb-4bit")


print(pipe(messages))

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-2-9b-it-bnb-4bit", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-9b-it-bnb-4bit")

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0]))

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Enable CPU offloading
# model = AutoModelForCausalLM.from_pretrained(
#     "unsloth/gemma-2-9b-it-bnb-4bit",
#     device_map="auto",
#     llm_int8_enable_fp32_cpu_offload=True,
#     torch_dtype=torch.float16
# )
# tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-9b-it-bnb-4bit")

# messages = [{"role": "user", "content": "Who are you?"}]
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
# outputs = model.generate(inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0]))


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig, pipeline

# model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total number of parameters: {total_params}')


# print(config.architectures)
# print("Supported classes:", config.auto_map)
# for name, param in model.named_parameters():
#     print(f'{name} : {param.shape}')
    
# for name, modul in model.named_modules():
#     print(f'{name} : {modul}')
    

# print(model.embed_tokens.weight)


# test = pipeline(task="sentiment-analysis", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# texts = [
#     "I absolutely loved this movie! The acting was fantastic and the story was captivating.",
#     "This product is terrible. It broke after just one use and the customer service was unhelpful.",
#     "The food at this restaurant was amazing, but the service was a bit slow.",
#     "I'm really disappointed with this purchase. The quality is much lower than I expected.",
#     "The conference was well-organized, and the speakers were very knowledgeable."
# ]


# result= test(texts)

# print(result)

# Cetak detail layer dan head
# for name, module in model.named_modules():
#     print(f"Layer: {name}")
#     print(f"  Type: {type(module)}")
#     print(f"  Parameters: {sum(p.numel() for p in module.parameters())}")
#     print(f"  Children: {list(module.children())}")
#     print("\n")

# Cetak output dari setiap layer dan head
# for name, module in model.named_modules():
#     if hasattr(module, 'output'):
#         print(f"Layer: {name}")
#         print(f"  Output: {module.output}")
#     if hasattr(module, 'attentions'):
#         print(f"  Attentions: {module.attentions}")
#     if hasattr(module, 'hidden_states'):
#         print(f"  Hidden States: {module.hidden_states}")
#     print("\n")
 
# from torchinfo import summary   
    
# print(model.forward)

# summary(model)