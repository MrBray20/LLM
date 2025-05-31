from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoModel, AutoTokenizer

model_path = "unsloth/gemma-2b-it-bnb-4bit"
path = "D:\SKRIPSI\Code Program\Fixmodel\model\Gemma"

model = AutoModelForCausalLM.from_pretrained(model_path)
model.save_pretrained(path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(path)
