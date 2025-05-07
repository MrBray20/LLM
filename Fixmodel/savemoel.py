from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B-Instruct",quantization_config=bnb_config)

# print(model.config.name_or_path) 

model.save_pretrained("D:\SKRIPSI\Code Program\Fixmodel\model\LLAMA", safe_serialization=True, max_shard_size="10GB")