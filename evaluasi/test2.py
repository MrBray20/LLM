import pandas as pd
import json
from collections import Counter
import re

def parse_prediction(pred_str):
    """
    Parses the prediction string (which is a string representation of a dictionary)
    into an actual Python dictionary.
    Handles potential issues like single quotes and missing keys gracefully.
    """
    try:
        # Replace single quotes with double quotes for valid JSON parsing
        # This is a common issue when parsing stringified Python dicts as JSON
        pred_str = pred_str.replace("'", '"')
        # Handle boolean values if any (though not present in your sample)
        pred_str = pred_str.replace("True", "true").replace("False", "false")
        return json.loads(pred_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e} for string: {pred_str}")
        return {} # Return an empty dict if parsing fails

def explore_explanations(df):
    """
    Explores the 'explanation' content from PredicMistral, PredicLLAMA, and PredicGemma.
    """
    print("--- Eksplorasi Penjelasan Model Sentimen ---")
    print("\nRingkasan Jumlah Penjelasan yang Berhasil Diurai:")

    for col in ['PredicMistral', 'PredicLLAMA', 'PredicGemma']:
        explanations = []
        for index, row in df.iterrows():
            pred_data = parse_prediction(row[col])
            if 'explanation' in pred_data:
                explanations.append(pred_data['explanation'])
        
        print(f"  {col}: {len(explanations)} penjelasan ditemukan.")
        
        if explanations:
            print(f"\n--- Analisis Kata Kunci Umum pada {col} ---")
            all_words = []
            for exp in explanations:
                # Tokenisasi kata, konversi ke lowercase, hapus non-alfabet
                words = re.findall(r'\b\w+\b', exp.lower())
                all_words.extend(words)
            
            # Filter stop words sederhana (bisa diperluas)
            stop_words = set(['the', 'a', 'an', 'is', 'it', 'text', 'contains', 'sentiment', 'because', 'which', 'of', 'and', 'to', 'in', 'that', 'with', 'due', 'for', 'or', 'as', 'by', 'from', 'has', 'have', 'had', 'its', 'not', 'no', 'but', 'can', 'do', 'does', 'did', 'are', 'was', 'were', 'be', 'been', 'being', 'this', 'that', 'these', 'those', 'just', 'only', 'so', 'such', 'very', 'more', 'most', 'some', 'any', 'all', 'etc', 'often', 'also', 'often', 'uses', 'use', 'using', 'indicates', 'indicated', 'expression', 'expressed', 'expresses'])
            
            filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
            
            word_counts = Counter(filtered_words)
            print(f"10 Kata Kunci Paling Sering Muncul:")
            for word, count in word_counts.most_common(10):
                print(f"  - {word}: {count}")

            print(f"\n--- Pencarian Frasa Spesifik pada {col} ---")
            
            # Anda bisa menambahkan frasa yang ingin Anda cari di sini
            search_phrases = ["sense of", "due to", "sarcasm", "negative emotion", "positive words", "negative words"]
            
            found_phrases_count = Counter()
            for exp in explanations:
                for phrase in search_phrases:
                    if phrase.lower() in exp.lower():
                        found_phrases_count[phrase] += 1
            
            if found_phrases_count:
                print("Frekuensi Frasa yang Dicari:")
                for phrase, count in found_phrases_count.items():
                    print(f"  - '{phrase}': {count} kali muncul")
            else:
                print("Tidak ada frasa spesifik yang ditemukan.")

            print("-" * 50)
    print("\nProgram selesai.")

# --- Data Anda dalam format DataFrame ---
# Membuat DataFrame dari data yang Anda berikan
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

# Jalankan program eksplorasi
explore_explanations(df)