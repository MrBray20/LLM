# from transformers import AutoConfig

# config = AutoConfig.from_pretrained("bert-base-uncased")
# print(config)

# from transformers import AutoModel

# model = AutoModel.from_pretrained("bert-base-uncased")
# print(model)

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokens = tokenizer("Hello, how are you?", return_tensors="tf")
# print(tokens)


# from transformers import pipeline
# from torch.utils.data import Dataset
# from tqdm.auto import tqdm

# pipe = pipeline("text-classification", device=0)


# class MyDataset(Dataset):
#     def __len__(self):
#         return 5000

#     def __getitem__(self, i):
#         return "This is a test"


# dataset = MyDataset()

# for batch_size in [1, 8, 64, 256]:
#     print("-" * 30)
#     print(f"Streaming batch_size={batch_size}")
#     for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
#         pass


from transformers import pipeline

# Load pipeline untuk sentiment analysis
classifier = pipeline("sentiment-analysis")

# Data input (beberapa kalimat)
texts = [
    "I love this product!",
    "This is the worst experience ever.",
    "Absolutely fantastic!",
    "Not bad, but could be better.",
    "Terrible service, I'm disappointed."
]

# Menjalankan pipeline dengan batch size = 2
results = classifier(texts, batch_size=2)

# Output hasil prediksi

print(results)
