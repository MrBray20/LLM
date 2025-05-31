from transformers import AutoTokenizer, AutoModel
import torch

# Pilih model yang sudah dilatih sebelumnya (misalnya, "bert-base-uncased")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tentukan teks yang ingin Anda ubah menjadi embedding
teks = "Saya suka makan apel."

# Tokenisasi teks
input_ids = tokenizer(teks, return_tensors="pt")

# Pastikan model dan input berada di perangkat yang sama (CPU atau GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_ids = input_ids.to(device)

# Nonaktifkan perhitungan gradien untuk efisiensi
with torch.no_grad():
    # Dapatkan output dari model
    output = model(**input_ids)

# Output berisi berbagai informasi, termasuk embedding kata
# Untuk model seperti BERT, embedding kata biasanya terdapat di output.last_hidden_state
embeddings = output.last_hidden_state

# `embeddings` sekarang berisi tensor dengan dimensi [batch_size, sequence_length, hidden_size]
# Dalam kasus ini, batch_size adalah 1, sequence_length adalah jumlah token, dan hidden_size adalah dimensi embedding (misalnya, 768 untuk bert-base-uncased)

# Anda dapat mengambil embedding untuk token tertentu
# Misalnya, untuk mendapatkan embedding kata pertama ("Saya"):
embedding_saya = embeddings[0, 0, :]

# Atau, untuk mendapatkan embedding untuk semua token:
# all_embeddings = embeddings[0]

# Cetak shape dari tensor embedding
print("Shape dari tensor embeddings:", embeddings.shape)

# Cetak 10 nilai pertama dari embedding kata "Saya"
print("10 nilai pertama dari embedding kata 'Saya':", embedding_saya[:10])

# Sekarang, mari kita coba dengan teks lain dan lihat bagaimana embeddingnya berbeda
teks2 = "Dia suka minum kopi."
input_ids2 = tokenizer(teks2, return_tensors="pt").to(device)
with torch.no_grad():
    output2 = model(**input_ids2)
embeddings2 = output2.last_hidden_state
embedding_dia = embeddings2[0, 0, :]

print("\nShape dari tensor embeddings2:", embeddings2.shape)
print("10 nilai pertama dari embedding kata 'Dia':", embedding_dia[:10])

# Kita dapat menghitung kesamaan antara embedding "Saya" dan "Dia"
from torch.nn.functional import cosine_similarity
similarity = cosine_similarity(embedding_saya.unsqueeze(0), embedding_dia.unsqueeze(0))
print(f"\nKesamaan kosinus antara 'Saya' dan 'Dia': {similarity.item():.4f}")

# Contoh lain untuk melihat perbedaan konteks
teks3 = "Apel adalah buah yang sehat."
input_ids3 = tokenizer(teks3, return_tensors="pt").to(device)
with torch.no_grad():
    output3 = model(**input_ids3)
embeddings3 = output3.last_hidden_state
embedding_apel3 = embeddings3[0, 1, :]  # Embedding untuk "Apel" dalam konteks ini

print("\nShape dari tensor embeddings3:", embeddings3.shape)
print("10 nilai pertama dari embedding kata 'Apel' (sebagai buah):", embedding_apel3[:10])

similarity_apel = cosine_similarity(embedding_saya.unsqueeze(0), embedding_apel3.unsqueeze(0))
print(f"\nKesamaan kosinus antara 'Saya' dan 'Apel' (buah): {similarity_apel.item():.4f}")

