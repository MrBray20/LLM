from transformers import GPT2Tokenizer, TFGPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('flax-community/gpt2-small-indonesian')
model = TFGPT2Model.from_pretrained('flax-community/gpt2-small-indonesian',from_pt=True)
text = "Ubah dengan teks apa saja."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(encoded_input)
