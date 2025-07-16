import pandas as pd
import ast # Untuk mengkonversi string kamus menjadi objek kamus
from collections import Counter
import re
# from IPython.display import display # Untuk tampilan DataFrame yang lebih baik - Dihapus karena ModuleNotFoundError

# --- 1. Persiapan Data (Sama seperti sebelumnya) ---
data = {
    'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'PredicMistral': ["{'sentiment': 'NEUTRAL', 'confidence': 0.95, 'explanation': 'The text does not contain any positive or negative words. It is a simple statement about the last session of the day.'}",
                      "{'sentiment': 'POSITIVE', 'confidence': 0.95, 'explanation': \"The text expresses positive sentiment towards Shanghai, as it is described as'really exciting' and 'good tweeps' are mentioned, which suggests positive interactions with people in China.\"}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 0.95, 'explanation': 'The text mentions that Veronique Branquinho had to quit her company due to a recession, which is generally considered a negative event.'}",
                      "{'sentiment': 'POSITIVE', 'confidence': 1.0, 'explanation': \"The text contains positive words such as 'happy' and 'bday', which are typically associated with positive sentiment.\"}",
                      "{'sentiment': 'POSITIVE', 'confidence': 1.0, 'explanation': \"The user explicitly states 'I like it!!', which is a positive sentiment.\"}",
                      "{'sentiment': 'POSITIVE', 'confidence': 1.0, 'explanation': \"The text contains exclamation marks and positive words like 'great' and 'visitors', indicating a positive sentiment.\"}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 0.99, 'explanation': \"The text contains words like 'hates' and 'here' which are generally associated with negative sentiment. The use of 'lol' might suggest sarcasm, but the overall context indicates a negative sentiment.\"}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 0.95, 'explanation': \"The user expresses a desire ('wish') that cannot be fulfilled ('but im in school and myspace is completely blocked'), indicating a negative sentiment.\"}",
                      "{'sentiment': 'NEUTRAL', 'confidence': 0.95, 'explanation': 'The text does not express any strong positive or negative emotions. It is a simple statement of fact.'}",
                      "{'sentiment': 'NEUTRAL', 'confidence': 0.8, 'explanation': \"The text contains a mix of positive and negative statements. The speaker mentions that their day is 'alright', which is a positive sentiment, but also mentions that they 'haven't done anything yet', which is a negative sentiment. The overall sentiment is neutral because the positive and negative sentiments are balanced.\"}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 0.95, 'explanation': \"The user expresses frustration and disappointment with the phrase 'total bummer', indicating a negative sentiment.\"}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 0.99, 'explanation': \"The text expresses disappointment or dissatisfaction as the speaker checked and found out that they didn't win.\"}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 0.9, 'explanation': 'The text contains a question that implies dissatisfaction or boredom with the tavern, which suggests a negative sentiment.'}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The user expresses feelings of sadness about their youngest son growing up, which indicates a negative sentiment.'}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 0.99, 'explanation': \"The text expresses frustration and dissatisfaction with the phone's condition, specifically mentioning that the socket is loose. This indicates a negative sentiment.\"}",
                      "{'sentiment': 'NEGATIVE', 'confidence': 1.0, 'explanation': \"The text contains negative words such as 'hot', 'don't like', 'hate', and 'bad'. The overall sentiment is negative.\"}"],
    'PredicLLAMA': ["{'sentiment': 'POSITIVE', 'confidence': 0.9, 'explanation': \"The text is positive because it mentions a 'last session of the day', which implies a sense of completion and satisfaction. The URL provided is also a link to a photo, which is often associated with a positive or uplifting experience.\"}",
                    "{'sentiment': 'POSITIVE', 'confidence': 0.9, 'explanation': \"The text expresses enthusiasm and excitement about Shanghai, using words like 'exciting' and'skyscrapers galore'. The tone is upbeat and positive, with a sense of admiration for the city's architecture.\"}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.9, 'explanation': \"The text expresses a negative sentiment towards the situation of Veronique Branquinho, as she has to quit her company due to recession. The use of the word'shame' emphasizes the negative emotional tone of the text.\"}",
                    "{'sentiment': 'POSITIVE', 'confidence': 0.9, 'explanation': \"contains enthusiastic language ('happy') and celebratory tone ('bday')\"}",
                    "{'sentiment': 'POSITIVE', 'confidence': 0.9, 'explanation': \"The text expresses enthusiasm and approval for the image shared, indicated by the use of 'I like it!!' which is a common expression of positive sentiment.\"}",
                    "{'sentiment': 'POSITIVE', 'confidence': 0.9, 'explanation': \"The text contains enthusiastic language, such as 'thats great', 'weee', and 'visitors!', which convey excitement and positivity.\"}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.9, 'explanation': \"The text expresses strong negative sentiment towards the author's perceived social status on the platform, indicated by the use of the phrase 'I THINK EVERYONE HATES ME' and the humorous tone of 'lol' which is often used to mask negative emotions.\"}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a desire that cannot be fulfilled due to circumstances (school and blocked MySpace), which creates a sense of frustration and disappointment.'}",
                    "{'sentiment': 'POSITIVE', 'confidence': 0.8, 'explanation': \"The text expresses a sense of accomplishment and efficiency, as it mentions 'a short time', implying that the solution was found quickly.\"}",
                    "{'sentiment': 'NEUTRAL', 'confidence': 0.5, 'explanation': 'The text expresses a neutral sentiment with a hint of frustration and boredom, but the tone is casual and conversational, which suggests a lack of strong emotions.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.9, 'explanation': \"The text expresses frustration and disappointment with the situation, using words like 'put on hold', 'total bummer', and 'argh' to convey a strong negative emotion.\"}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': \"The text expresses disappointment and frustration, as indicated by the phrase 'we didn't win', which is a negative outcome. The tone is also somewhat apologetic, as the speaker is acknowledging their failure to check the result.\"}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': \"The text contains a sarcastic tone, with the use of 'tavern bore' which is a colloquialism for someone who is boring, and the question mark at the end of the sentence, which implies annoyance or frustration.\"}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': \"The text expresses a mix of emotions, but the dominant sentiment is sadness, as indicated by the phrase 'it makes me kinda sad'. The mention of the child's growth and the use of the word'sad' further reinforce this sentiment.\"}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.9, 'explanation': 'The text contains profanity, sarcasm, and frustration, which are all negative emotions.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.9, 'explanation': 'The text expresses strong negative emotions towards the weather and a personal situation, using emoticons to emphasize the negativity.'}"],
    'PredicGemma': ["{'sentiment': 'NEUTRAL', 'confidence': 0.5, 'explanation': 'The text provides a link but does not express any strong positive or negative sentiment.'}",
                    "{'sentiment': 'POSITIVE', 'confidence': 0.8, 'explanation': 'The text expresses a positive sentiment towards Shanghai and China.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': \"The text expresses a negative sentiment towards Veronique Branquinho's situation.\"}",
                    "{'sentiment': 'POSITIVE', 'confidence': 1, 'explanation': 'The text expresses a positive sentiment.'}",
                    "{'sentiment': 'POSITIVE', 'confidence': 0.95, 'explanation': 'The text expresses a positive sentiment.'}",
                    "{'sentiment': 'POSITIVE', 'confidence': 1, 'explanation': 'The text expresses a positive sentiment.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.2, 'explanation': 'The text expresses a negative sentiment, with the use of sarcasm and a negative tone.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a sense of disappointment and frustration due to being unable to access social media.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a sense of regret and disappointment.'}",
                    "{'sentiment': 'Neutral', 'confidence': 0.5, 'explanation': 'The text is neutral and does not express a strong positive or negative sentiment.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a sense of disappointment and frustration.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a sense of disappointment and loss.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a sense of dissatisfaction and boredom.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.7, 'explanation': 'The text expresses a sense of sadness and concern about the growth and maturity of the child.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a sense of dissatisfaction and frustration.'}",
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a negative sentiment, with the speaker expressing dislike and dissatisfaction.'}"],
}

df = pd.DataFrame(data)

# Fungsi untuk mengekstrak 'explanation' dan 'sentiment'
def extract_info(text_str):
    try:
        data_dict = ast.literal_eval(text_str)
        return data_dict.get('explanation'), data_dict.get('sentiment')
    except (ValueError, SyntaxError, KeyError):
        return None, None

# Menerapkan fungsi ke setiap kolom Predic
df[['Mistral_Explanation', 'Mistral_Sentiment']] = df['PredicMistral'].apply(lambda x: pd.Series(extract_info(x)))
df[['LLAMA_Explanation', 'LLAMA_Sentiment']] = df['PredicLLAMA'].apply(lambda x: pd.Series(extract_info(x)))
df[['Gemma_Explanation', 'Gemma_Sentiment']] = df['PredicGemma'].apply(lambda x: pd.Series(extract_info(x)))

print("DataFrame dengan kolom penjelasan dan sentimen yang diekstrak:")
print(df[['index', 'Mistral_Sentiment', 'Mistral_Explanation', 'LLAMA_Sentiment', 'LLAMA_Explanation', 'Gemma_Sentiment', 'Gemma_Explanation']].head().to_string()) # Menggunakan .to_string() untuk tampilan penuh

# --- 2. Analisis Kata Kunci Spesifik Sentimen per Model ---

def get_sentiment_specific_keywords(df_column_explanation, df_column_sentiment, sentiment_type, n=10):
    """
    Mengambil kata kunci teratas dari penjelasan untuk sentimen tertentu.
    """
    corpus = df_column_explanation[df_column_sentiment == sentiment_type].dropna()
    words = re.findall(r'\b\w+\b', ' '.join(corpus).lower())
    # Stopwords yang lebih komprehensif
    stopwords = set([
        'the', 'text', 'is', 'a', 'of', 'and', 'to', 'in', 'it', 'sentiment', 'with', 'due', 'for', 'that', 'as', 'on', 'or', 'an', 'has', 'not', 'no', 'but', 'by', 'at', 'from', 'be', 'are', 'which', 'this', 'its', 'their', 'expresses',
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'couldn', "couldn't", 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'also', 'can', 'like', 'one', 'use', 'using', 'common', 'indicated', 'implies', 'convey', 'sense', 'tone', 'words', 'phrases', 'contains', 'generally', 'often', 'due', 'makes', 'much', 'strong', 'simple', 'statement', 'about', 'last', 'day', 'url', 'provided', 'photo', 'associated', 'uplifting', 'experience', 'described', 'real', 'good', 'suggests', 'interactions', 'people', 'china', 'enthusiasm', 'excitement', 'shanghai', 'skyscrapers', 'galore', 'upbeat', 'admiration', 'city', 'architecture', 'situation', 'branquinho', 'quit', 'company', 'recession', 'considered', 'event', 'shame', 'emphasizes', 'emotional', 'happy', 'bday', 'typically', 'associated', 'explicitly', 'states', 'i', 'it', 'common', 'expression', 'exclamation', 'marks', 'great', 'visitors', 'indicating', 'convey', 'excitement', 'positivity', 'strong', 'author', 'perceived', 'social', 'status', 'platform', 'phrase', 'think', 'everyone', 'hates', 'me', 'humorous', 'mask', 'desire', 'cannot', 'fulfilled', 'circumstances', 'school', 'blocked', 'myspace', 'creates', 'frustration', 'disappointment', 'emotions', 'fact', 'mix', 'statements', 'speaker', 'alright', 'done', 'yet', 'overall', 'balanced', 'hint', 'boredom', 'casual', 'conversational', 'lack', 'total', 'bummer', 'argh', 'disappointment', 'dissatisfaction', 'checked', 'found', 'didn', 'win', 'outcome', 'somewhat', 'apologetic', 'acknowledging', 'failure', 'check', 'result', 'sarcastic', 'tavern', 'bore', 'colloquialism', 'boring', 'question', 'mark', 'end', 'sentence', 'annoyance', 'feelings', 'sadness', 'youngest', 'son', 'growing', 'child', 'growth', 'maturity', 'phone', 'condition', 'specifically', 'mentioning', 'socket', 'loose', 'profanity', 'emoticons', 'emphasize', 'hot', 'dont', 'hate', 'bad', 'weather', 'personal'
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


# --- 3. Perbandingan Penjelasan Berdampingan untuk Index Tertentu ---

def compare_explanations_for_index(dataframe, target_index):
    """
    Menampilkan penjelasan dari ketiga model untuk indeks tertentu.
    """
    row = dataframe[dataframe['index'] == target_index]
    if not row.empty:
        row = row.iloc[0]
        print(f"\n--- Perbandingan Penjelasan untuk Index: {target_index} ---")
        print(f"  Mistral Pred: {row['Mistral_Sentiment']}, Expl: {row['Mistral_Explanation']}")
        print(f"  LLAMA Pred:   {row['LLAMA_Sentiment']}, Expl: {row['LLAMA_Explanation']}")
        print(f"  Gemma Pred:   {row['Gemma_Sentiment']}, Expl: {row['Gemma_Explanation']}")
    else:
        print(f"Index {target_index} tidak ditemukan dalam DataFrame.")

# Contoh penggunaan: Bandingkan penjelasan untuk indeks 0 (kasus netral/positif yang berbeda)
compare_explanations_for_index(df, 0)

# Contoh penggunaan: Bandingkan penjelasan untuk indeks 6 (kasus negatif dengan sarkasme)
compare_explanations_for_index(df, 6)

# Contoh penggunaan: Bandingkan penjelasan untuk indeks 8 (kasus netral/positif/negatif yang berbeda)
compare_explanations_for_index(df, 8)

# --- 4. Visualisasi (Konseptual: Word Clouds) ---
print("\n--- Visualisasi Konseptual: Word Clouds ---")
print("Untuk visualisasi yang lebih intuitif, Anda dapat membuat 'Word Clouds' dari penjelasan setiap model atau sentimen.")
print("Ini akan menampilkan kata-kata yang paling sering muncul dalam ukuran yang lebih besar.")
print("Anda akan membutuhkan library seperti `wordcloud` dan `matplotlib`.")
print("\nContoh kode konseptual untuk membuat word cloud (tidak dijalankan di sini):")
print("""
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # Gabungkan semua penjelasan Mistral untuk sentimen Positif
# text_mistral_positive = " ".join(df['Mistral_Explanation'][df['Mistral_Sentiment'] == 'POSITIVE'].dropna())

# # Buat objek WordCloud
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_mistral_positive)

# # Tampilkan gambar
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud Penjelasan Mistral (Sentimen Positif)')
# plt.show()
""")
print("Anda bisa mengulang proses ini untuk setiap model dan setiap jenis sentimen.")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Gabungkan semua penjelasan Mistral untuk sentimen Positif
text_mistral_positive = " ".join(df['Mistral_Explanation'][df['Mistral_Sentiment'] == 'POSITIVE'].dropna())

# Buat objek WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_mistral_positive)

# Tampilkan gambar
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud Penjelasan Mistral (Sentimen Positif)')
plt.show()