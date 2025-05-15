from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import ast


def extrack_sentiment(json_str):
    try:
        parsed = ast.literal_eval(json_str)
        return parsed.get('sentiment',None)
    except:
        return None
    
def extract_sentiments(row):
    return pd.Series({
        'sentiment_mistral': extrack_sentiment(row['PredicMistral']).lower(),
        'sentiment_llama':extrack_sentiment(row['PredicLLAMA']).lower(),
        'sentiment_gemma':extrack_sentiment(row['PredicGemma']).lower(),
    })


def evaluate_sentiment_analysis(y_true, y_pred, labels=['positive', 'neutral', 'negative']):
    """
    Fungsi untuk mengevaluasi model analisis sentimen
    
    Parameters:
    -----------
    y_true : array-like
        Label sebenarnya
    y_pred : array-like
        Label prediksi
    labels : list
        Daftar label yang digunakan
        
    Returns:
    --------
    dict
        Dictionary berisi berbagai metrik evaluasi
    """
    #Check Unique
    print(y_pred.unique())
    
    # Confusion Matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Laporan Klasifikasi
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    # Akurasi
    acc = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    
    # Print hasil evaluasi
    print("------- Hasil Evaluasi Model Analisis Sentimen -------")
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels))
    
    print(f"\nAkurasi: {acc:.4f}")
    
    print("\nMetrik per Kelas:")
    for i, label in enumerate(labels):
        print(f"{label}:")
        print(f"  - Precision: {prec[i]:.4f}")
        print(f"  - Recall: {rec[i]:.4f}")
        print(f"  - F1-Score: {f1[i]:.4f}")
        print(f"  - Support: {sup[i]}")
        
    # Return hasil evaluasi
    return {
        'confusion_matrix': conf_mat,
        'classification_report': report,
        'accuracy': acc,
        'metrics_by_class': pd.DataFrame({
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Support': sup
        }, index=labels)
    }
def proses_pandas(data):
    res = data[['PredicMistral','PredicLLAMA','PredicGemma']].apply(extract_sentiments,axis=1)   
    res = pd.concat([data,res],axis=1)
    return res 

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
test1 = pd.read_csv(r"D:\SKRIPSI\Code Program\evaluasi\test1.csv", delimiter=',')
test1 = proses_pandas(test1)
print(test1)
test1_mistral = test1['sentiment_mistral']
test1_llama = test1['sentiment_llama']
test1_gemma = test1['sentiment_gemma']

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

test2 = pd.read_csv(r"D:\SKRIPSI\Code Program\evaluasi\test2.csv", delimiter=',')
test2 = proses_pandas(test2)

test2_mistral = test2['sentiment_mistral']
test2_llama = test2['sentiment_llama']
test2_gemma = test2['sentiment_gemma']


test3 = pd.read_csv(r"D:\SKRIPSI\Code Program\evaluasi\test3.csv", delimiter=',')
test3 = proses_pandas(test3)

test3_mistral = test3['sentiment_mistral']
test3_llama = test3['sentiment_llama']
test3_gemma = test3['sentiment_gemma']


no_index = test1['index']
datatest_predic = pd.read_csv(r"D:\SKRIPSI\Code Program\Fixmodel\archive\test.csv").iloc[no_index]['sentiment']

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hasil Mistral >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("------------------------------------------- Test 1 Mistral -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test1_mistral)
print("------------------------------------------- Test 2 Mistral -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test2_mistral)
print("------------------------------------------- Test 3 Mistral -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test3_mistral)


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hasil LLAMA >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("------------------------------------------- Test 1 LLAMA -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test1_llama)
print("------------------------------------------- Test 2 LLAMA -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test2_llama)
print("------------------------------------------- Test 3 LLAMA -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test3_llama)



print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hasil Gemma >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("------------------------------------------- Test 1 Gemma -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test1_gemma)
print("------------------------------------------- Test 2 Gemma -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test2_gemma)
print("------------------------------------------- Test 3 Gemma -------------------------------------------")
evaluate_sentiment_analysis(datatest_predic,test3_gemma)

