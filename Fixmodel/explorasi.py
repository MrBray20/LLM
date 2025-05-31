import pandas as pd


# print("Kolom pada data")
# print(df)
# for col in df.columns:
#     print(col)


df = pd.read_csv(r"D:\SKRIPSI\Code Program\Fixmodel\archive\test.csv", encoding="latin1")
df = df.dropna(subset=['text'])
dftest = df[["text","sentiment"]]


# ==============================
# 1. Menghitung jumlah kata di kolom 'text'
# ==============================
# Tambahkan kolom baru 'word_count'
dftest['word_count'] = dftest['text'].apply(lambda x: len(str(x).split()))

# ==============================
# 2. Melihat label unik pada kolom 'sentiment'
# ==============================
print("Label Sentimen Unik:")
print(dftest['sentiment'].unique())

# ==============================
# 3. Menghitung persebaran jumlah label 'sentiment'
# ==============================
print("\nPersebaran Sentimen:")
print(dftest['sentiment'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Hitung persebaran sentimen
sentiment_counts = dftest['sentiment'].value_counts()

# -----------------------------
# Bar Chart
# -----------------------------
plt.figure(figsize=(8,5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title("Persebaran Label Sentimen")
plt.xlabel("Label Sentimen")
plt.ylabel("Jumlah")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# # -----------------------------
# # Pie Chart
# # -----------------------------
# plt.figure(figsize=(6,6))
# plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
# plt.title("Proporsi Label Sentimen")
# plt.show()
#


# Menghitung jumlah kata di kolom 'text'
dftest['word_count'] = dftest['text'].apply(lambda x: len(str(x).split()))

# Hitung frekuensi masing-masing jumlah kata
word_count_dist = dftest['word_count'].value_counts().sort_index()

# Buat grafik batang
plt.figure(figsize=(12,6))
sns.barplot(x=word_count_dist.index, y=word_count_dist.values, color='skyblue')
plt.title("Distribusi Jumlah Kata per Teks")
plt.xlabel("Jumlah Kata")
plt.ylabel("Jumlah Teks")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Hitung jumlah kata per baris
# dftest['word_count'] = dftest['text'].apply(lambda x: len(str(x).split()))

# Hitung frekuensi masing-masing word count
word_count_freq = dftest['word_count'].value_counts().sort_index()

# Cari word count paling umum
most_common_word_count = word_count_freq.idxmax()
most_common_count = word_count_freq.max()

print(f"Jumlah kata yang paling banyak muncul dalam teks adalah: {most_common_word_count} kata")
print(f"Jumlah teks yang memiliki {most_common_word_count} kata: {most_common_count}")
