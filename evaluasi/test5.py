import pandas as pd
import ast # Untuk mengkonversi string kamus menjadi objek kamus
from collections import Counter
import re
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 

df = pd.read_csv(r"D:\SKRIPSI\Code Program\evaluasi\test1.csv", delimiter=',')

# Fungsi untuk mengekstrak 'explanation' dan 'sentiment' dari string dictionary
def extract_info(text_str):
    try:
        data_dict = ast.literal_eval(text_str)
        return data_dict.get('explanation'), data_dict.get('sentiment')
    except (ValueError, SyntaxError, KeyError):
        return None, None

# Menerapkan fungsi ke setiap kolom Predic untuk mengekstrak penjelasan dan sentimen
df[['Mistral_Explanation', 'Mistral_Sentiment']] = df['PredicMistral'].apply(lambda x: pd.Series(extract_info(x)))
df[['LLAMA_Explanation', 'LLAMA_Sentiment']] = df['PredicLLAMA'].apply(lambda x: pd.Series(extract_info(x)))
df[['Gemma_Explanation', 'Gemma_Sentiment']] = df['PredicGemma'].apply(lambda x: pd.Series(extract_info(x)))

print("DataFrame dengan kolom penjelasan dan sentimen yang diekstrak:")
print(df[['index', 'Mistral_Sentiment', 'Mistral_Explanation', 'LLAMA_Sentiment', 'LLAMA_Explanation', 'Gemma_Sentiment', 'Gemma_Explanation']].head().to_string())

# --- 2. Analisis Kata Kunci Spesifik Sentimen per Model ---

def get_sentiment_specific_keywords(df_column_explanation, df_column_sentiment, sentiment_type, n=10):


    corpus = df_column_explanation[df_column_sentiment == sentiment_type].dropna()

    words = re.findall(r'\b\w+\b', ' '.join(corpus).lower())


    stopwords = set([
        'the', 'text', 'is', 'a', 'of', 'and', 'to', 'in', 'it', 'sentiment', 'with', 'due', 'for', 'that', 'as', 'on', 'or', 'an', 
        'has', 'not', 'no', 'but', 'by', 'at', 'from', 'be', 'are', 'which', 'this', 'its', 'their', 'expresses',
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 
        'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'couldn', 
        "couldn't", 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 
        'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 
        'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', 
        'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', 
        "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 
        'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 
        'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 
        'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 
        'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 
        'yourself', 'yourselves', 'also', 'can', 'like', 'one', 'use', 'using', 'common', 'indicated', 'implies', 'convey', 'sense', 'tone', 
        'words', 'phrases', 'contains', 'generally', 'often', 'due', 'makes', 'much', 'strong', 'simple', 'statement', 'about', 'last', 'day', 
        'url', 'provided', 'photo', 'associated', 'uplifting', 'experience', 'described', 'real', 'good', 'suggests', 'interactions', 'people', 
        'china', 'enthusiasm', 'excitement', 'shanghai', 'skyscrapers', 'galore', 'upbeat', 'admiration', 'city', 'architecture', 'situation', 
        'branquinho', 'quit', 'company', 'recession', 'considered', 'event', 'shame', 'emphasizes', 'emotional', 'happy', 'bday', 'typically', 
        'associated', 'explicitly', 'states', 'i', 'it', 'common', 'expression', 'exclamation', 'marks', 'great', 'visitors', 'indicating', 'convey', 
        'excitement', 'positivity', 'strong', 'author', 'perceived', 'social', 'status', 'platform', 'phrase', 'think', 'everyone', 'hates', 'me', 
        'humorous', 'mask', 'desire', 'cannot', 'fulfilled', 'circumstances', 'school', 'blocked', 'myspace', 'creates', 'frustration', 'disappointment', 
        'emotions', 'fact', 'mix', 'statements', 'speaker', 'alright', 'done', 'yet', 'overall', 'balanced', 'hint', 'boredom', 'casual', 
        'conversational', 'lack', 'total', 'bummer', 'argh', 'disappointment', 'dissatisfaction', 'checked', 'found', 'didn', 'win', 'outcome', 
        'somewhat', 'apologetic', 'acknowledging', 'failure', 'check', 'result', 'sarcastic', 'tavern', 'bore', 'colloquialism', 'boring', 'question', 
        'mark', 'end', 'sentence', 'annoyance', 'feelings', 'sadness', 'youngest', 'son', 'growing', 'child', 'growth', 'maturity', 'phone', 
        'condition', 'specifically', 'mentioning', 'socket', 'loose', 'profanity', 'emoticons', 'emphasize', 'hot', 'dont', 'hate', 'bad', 'weather', 
        'personal'
    ])

    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]

    return Counter(filtered_words).most_common(n)

print("\n--- Kata Kunci Teratas dalam Penjelasan Mistral (POSITIF) ---")
print(get_sentiment_specific_keywords(df['Mistral_Explanation'], df['Mistral_Sentiment'], 'POSITIVE'))

print("\n--- Kata Kunci Teratas dalam Penjelasan Mistral (NEGATIF) ---")
print(get_sentiment_specific_keywords(df['Mistral_Explanation'], df['Mistral_Sentiment'], 'NEGATIVE'))

print("\n--- Kata Kunci Teratas dalam Penjelasan Mistral (NETRAL) ---")
print(get_sentiment_specific_keywords(df['Mistral_Explanation'], df['Mistral_Sentiment'], 'NEUTRAL'))

print("\n--- Kata Kunci Teratas dalam Penjelasan LLAMA (POSITIF) ---")
print(get_sentiment_specific_keywords(df['LLAMA_Explanation'], df['LLAMA_Sentiment'], 'POSITIVE'))

print("\n--- Kata Kunci Teratas dalam Penjelasan LLAMA (NEGATIF) ---")
print(get_sentiment_specific_keywords(df['LLAMA_Explanation'], df['LLAMA_Sentiment'], 'NEGATIVE'))

print("\n--- Kata Kunci Teratas dalam Penjelasan LLAMA (NETRAL) ---")
print(get_sentiment_specific_keywords(df['LLAMA_Explanation'], df['LLAMA_Sentiment'], 'NEUTRAL'))

print("\n--- Kata Kunci Teratas dalam Penjelasan Gemma (POSITIF) ---")
print(get_sentiment_specific_keywords(df['Gemma_Explanation'], df['Gemma_Sentiment'], 'POSITIVE'))

print("\n--- Kata Kunci Teratas dalam Penjelasan Gemma (NEGATIF) ---")
print(get_sentiment_specific_keywords(df['Gemma_Explanation'], df['Gemma_Sentiment'], 'NEGATIVE'))

print("\n--- Kata Kunci Teratas dalam Penjelasan Gemma (NETRAL) ---")
print(get_sentiment_specific_keywords(df['Gemma_Explanation'], df['Gemma_Sentiment'], 'NEUTRAL'))



def compare_explanations_for_index(dataframe, target_index):
    """
    Menampilkan penjelasan dari ketiga model untuk indeks tertentu.
    Parameter:
    - dataframe: DataFrame yang berisi data sentimen dan penjelasan.
    - target_index: Nilai 'index' dari baris yang ingin dibandingkan.
    """
    row = dataframe[dataframe['index'] == target_index]
    if not row.empty:
        row = row.iloc[0] # Ambil baris pertama jika ada beberapa dengan indeks yang sama
        print(f"\n--- Perbandingan Penjelasan untuk Index: {target_index} ---")
        print(f"  Prediksi Mistral: {row['Mistral_Sentiment']}, Penjelasan: {row['Mistral_Explanation']}")
        print(f"  Prediksi LLAMA:   {row['LLAMA_Sentiment']}, Penjelasan: {row['LLAMA_Explanation']}")
        print(f"  Prediksi Gemma:   {row['Gemma_Sentiment']}, Penjelasan: {row['Gemma_Explanation']}")
    else:
        print(f"Index {target_index} tidak ditemukan dalam DataFrame.")


compare_explanations_for_index(df, 0)

compare_explanations_for_index(df, 6)


compare_explanations_for_index(df, 8)


print("\n--- Visualisasi: Word Clouds ---")
print("Membuat dan menampilkan Word Clouds untuk setiap model dan jenis sentimen.")

models = ['Mistral', 'LLAMA', 'Gemma']
sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

for model in models:
    for sentiment in sentiments:
        # Gabungkan semua penjelasan untuk model dan sentimen tertentu
        text_corpus = " ".join(df[f'{model}_Explanation'][df[f'{model}_Sentiment'] == sentiment].dropna())

        if text_corpus: # Pastikan ada teks untuk membuat word cloud
            # Buat objek WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  stopwords=get_sentiment_specific_keywords(
                                      df[f'{model}_Explanation'], df[f'{model}_Sentiment'], sentiment, n=0
                                  )).generate(text_corpus)

            # Tampilkan gambar
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off') # Sembunyikan sumbu
            plt.title(f'Word Cloud Penjelasan {model} (Sentimen {sentiment})')
            plt.show()
        else:
            print(f"Tidak ada data penjelasan untuk {model} (Sentimen {sentiment}) untuk membuat Word Cloud.")

print("\nProses pembuatan Word Cloud selesai.")