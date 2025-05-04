import evaluate

# Referensi (ground truth)
references = ["The quick brown fox jumps over the lazy dog."]

# Prediksi dari model
predictions = ["A quick brown fox jumps over the lazy dog."]

# --- BLEU ---
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=predictions, references=[references])
print("BLEU Score:", bleu_result['bleu'])

# --- ROUGE ---
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=predictions, references=references)
print("ROUGE Scores:", rouge_result)

# --- METEOR ---
meteor = evaluate.load("meteor")
meteor_result = meteor.compute(predictions=predictions, references=references)
print("METEOR Score:", meteor_result['meteor'])

# --- BERTScore (butuh internet untuk download model pertama kali) ---
bertscore = evaluate.load("bertscore")
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
print("BERTScore (F1):", sum(bertscore_result['f1']) / len(bertscore_result['f1']))
