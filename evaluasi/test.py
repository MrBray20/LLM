import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Contoh data (dalam praktiknya, ganti dengan data Anda sendiri)
# Anggap kita memiliki hasil prediksi dan nilai sebenarnya
y_true = ['positive', 'negative', 'neutral', 'positive', 'negative', 
          'neutral', 'positive', 'negative', 'neutral', 'positive',
          'negative', 'neutral', 'positive', 'negative', 'neutral']

y_pred = ['positive', 'negative', 'neutral', 'positive', 'negative', 
          'positive', 'neutral', 'negative', 'negative', 'positive',
          'neutral', 'neutral', 'positive', 'positive', 'neutral']

# Kategori sentimen
labels = ['positive', 'neutral', 'negative']

# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
print("Confusion Matrix:")
print(conf_matrix)

# 2. Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix untuk Analisis Sentimen')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 3. Laporan Klasifikasi
class_report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
df_report = pd.DataFrame(class_report).transpose()
print("\nLaporan Klasifikasi:")
print(df_report.round(3))

# 4. Akurasi keseluruhan
accuracy = accuracy_score(y_true, y_pred)
print(f"\nAkurasi Keseluruhan: {accuracy:.3f}")

# 5. Precision, Recall, F1-Score untuk setiap kelas
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels)

# Tampilkan sebagai DataFrame untuk memudahkan pembacaan
eval_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
}, index=labels)

print("\nMetrik Evaluasi per Kelas:")
print(eval_df.round(3))

# 6. Visualisasi Metrik per Kelas
plt.figure(figsize=(10, 6))
eval_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar')
plt.title('Metrik Evaluasi untuk Setiap Kelas Sentimen')
plt.ylabel('Skor')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('evaluation_metrics.png')
plt.show()

# 7. Bonus: Membuat Fungsi untuk Evaluasi yang Mudah Digunakan

def evaluate_sentiment_analysis(y_true, y_pred, labels=['positive', 'neutral', 'negative'], 
                               save_plots=True, plot_path_prefix=''):
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
    save_plots : bool
        Apakah menyimpan plot sebagai file gambar
    plot_path_prefix : str
        Awalan untuk path file gambar yang disimpan
        
    Returns:
    --------
    dict
        Dictionary berisi berbagai metrik evaluasi
    """
    # Confusion Matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Laporan Klasifikasi
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    # Akurasi
    acc = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    
    # Visualisasi Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix untuk Analisis Sentimen')
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{plot_path_prefix}confusion_matrix.png")
    plt.show()
    
    # Visualisasi Metrik per Kelas
    eval_metrics = pd.DataFrame({
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Support': sup
    }, index=labels)
    
    plt.figure(figsize=(10, 6))
    eval_metrics[['Precision', 'Recall', 'F1-Score']].plot(kind='bar')
    plt.title('Metrik Evaluasi untuk Setiap Kelas Sentimen')
    plt.ylabel('Skor')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{plot_path_prefix}evaluation_metrics.png")
    plt.show()
    
    # Return hasil evaluasi
    return {
        'confusion_matrix': conf_mat,
        'classification_report': report,
        'accuracy': acc,
        'metrics_by_class': eval_metrics
    }

# Contoh penggunaan fungsi evaluasi
print("\nMenggunakan Fungsi Evaluasi:")
results = evaluate_sentiment_analysis(y_true, y_pred)