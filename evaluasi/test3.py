import pandas as pd
import json
from collections import Counter
import re

# --- 1. Fungsi untuk Mengurai String Prediksi ---
def parse_prediction(pred_str):
    """
    Mengurai string prediksi (yang merupakan representasi string dari dictionary)
    menjadi dictionary Python yang sebenarnya.
    Menangani masalah umum seperti kutip tunggal dan nilai boolean yang salah.
    """
    try:
        # Ganti kutip tunggal dengan kutip ganda agar valid untuk parsing JSON
        pred_str = pred_str.replace("'", '"')
        # Ganti nilai boolean Python ke format JSON (True -> true, False -> false)
        pred_str = pred_str.replace("True", "true").replace("False", "false")
        return json.loads(pred_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e} for string: {pred_str}")
        return {} # Mengembalikan dictionary kosong jika parsing gagal

# --- 2. Data Anda (Representasi DataFrame) ---
# Data yang Anda berikan, diubah ke format dictionary untuk DataFrame
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
                    "{'sentiment': 'POSITIVE', 'confidence': 0.9, 'explanation': \"The text contains enthusiastic language, such as 'that`s great', 'weee', and 'visitors!', which convey excitement and positivity.\"}",
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
                    "{'sentiment': 'NEGATIVE', 'confidence': 0.8, 'explanation': 'The text expresses a negative sentiment, with the speaker expressing dislike and dissatisfaction.'}"
    ]
}

df = pd.DataFrame(data)

# --- 3. Kumpulan Stop Words untuk Pembersihan Teks ---
# Anda bisa memperluas daftar ini sesuai kebutuhan
stop_words = set([
    'the', 'a', 'an', 'is', 'it', 'text', 'contains', 'sentiment', 'because', 'which',
    'of', 'and', 'to', 'in', 'that', 'with', 'due', 'for', 'or', 'as', 'by', 'from',
    'has', 'have', 'had', 'its', 'not', 'no', 'but', 'can', 'do', 'does', 'did',
    'are', 'was', 'were', 'be', 'been', 'being', 'this', 'that', 'these', 'those',
    'just', 'only', 'so', 'such', 'very', 'more', 'most', 'some', 'any', 'all',
    'etc', 'often', 'also', 'often', 'uses', 'use', 'using', 'indicates', 'indicated',
    'expression', 'expressed', 'expresses', 'a', 'an', 'the', 'is', 'am', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'shall', 'should', 'may', 'might', 'can', 'could', 'must', 'and', 'but', 'or',
    'as', 'if', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
    'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
])

# --- 4. Fungsi Utama untuk Eksplorasi Penjelasan ---
def analyze_model_explanations(df, models_to_analyze, common_search_phrases=None):
    """
    Menganalisis penjelasan dari model-model yang ditentukan dalam DataFrame.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data sentimen.
        models_to_analyze (list): Daftar nama kolom model untuk dianalisis
                                   (misalnya, ['PredicMistral', 'PredicLLAMA']).
        common_search_phrases (list, optional): Daftar frasa yang ingin dicari
                                                 di semua penjelasan model.
    """
    if common_search_phrases is None:
        common_search_phrases = [
            "sarcasm", "negative emotion", "positive words", "negative words",
            "sense of", "due to", "implies", "frustration", "disappointment",
            "accomplishment", "sadness", "boredom", "neutral sentiment"
        ]

    results = {}

    for col in models_to_analyze:
        print(f"\n--- Analisis untuk Model: {col} ---")
        model_explanations = []
        model_sentiments = []

        # Ekstraksi Penjelasan dan Sentimen
        for index, row in df.iterrows():
            pred_data = parse_prediction(row[col])
            if 'explanation' in pred_data:
                model_explanations.append(pred_data['explanation'])
                model_sentiments.append(pred_data.get('sentiment', 'UNKNOWN')) # Ambil sentimen juga

        print(f"  Jumlah penjelasan ditemukan: {len(model_explanations)}")

        if not model_explanations:
            print("  Tidak ada penjelasan untuk dianalisis.")
            results[col] = {}
            continue

        # Analisis Kata Kunci Umum
        all_words = []
        for exp in model_explanations:
            # Tokenisasi: pisahkan kata, konversi ke lowercase, hapus non-alfabet
            words = re.findall(r'\b\w+\b', exp.lower())
            all_words.extend(words)

        filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
        word_counts = Counter(filtered_words)
        print("\n  10 Kata Kunci Paling Sering Muncul:")
        for word, count in word_counts.most_common(10):
            print(f"    - {word}: {count}")
        results[col] = {"top_words": word_counts.most_common(10)}

        # Analisis Frekuensi Frasa Spesifik
        found_phrases_count = Counter()
        for exp in model_explanations:
            for phrase in common_search_phrases:
                if phrase.lower() in exp.lower():
                    found_phrases_count[phrase] += 1

        if found_phrases_count:
            print("\n  Frekuensi Frasa Kunci dalam Penjelasan:")
            for phrase, count in found_phrases_count.items():
                print(f"    - '{phrase}': {count} kali muncul")
            results[col]["search_phrases"] = found_phrases_count
        else:
            print("  Tidak ada frasa spesifik yang ditemukan.")

        # Contoh: Menemukan Penjelasan untuk Sentimen Negatif yang Menyebut 'sarcasm'
        print("\n  Contoh Penjelasan Sentimen Negatif yang Menyebut 'sarcasm':")
        sarcasm_negative_explanations = [
            (i, exp) for i, (exp, sent) in enumerate(zip(model_explanations, model_sentiments))
            if 'sarcasm' in exp.lower() and sent.lower() == 'negative'
        ]
        if sarcasm_negative_explanations:
            for idx, exp_text in sarcasm_negative_explanations[:3]: # Tampilkan hingga 3 contoh
                print(f"    Index Data Asli: {idx} - \"{exp_text[:100]}...\"")
        else:
            print("    Tidak ditemukan contoh.")
        
        print("-" * 60)
    
    return results

# --- 5. Jalankan Analisis ---
models_to_analyze = ['PredicMistral', 'PredicLLAMA', 'PredicGemma']
analysis_results = analyze_model_explanations(df, models_to_analyze)

print("\n--- Ringkasan Hasil Analisis ---")
# Anda bisa memproses 'analysis_results' di sini untuk perbandingan lebih lanjut
for model, data in analysis_results.items():
    print(f"\nModel: {model}")
    if "top_words" in data:
        print("  Top Words:", data["top_words"])
    if "search_phrases" in data:
        print("  Search Phrases Counts:", data["search_phrases"])