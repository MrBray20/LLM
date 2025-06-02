from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

modelName = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizerManual = AutoTokenizer.from_pretrained(modelName)
modelManual = AutoModelForSequenceClassification.from_pretrained(modelName)

textToAnalyze = "Hugging Face Transformers sangat membantu!"

inputsManual = tokenizerManual(textToAnalyze, return_tensors="pt") # "pt" untuk PyTorch tensors

with torch.no_grad():
    outputsManual = modelManual(**inputsManual)

logitsManual = outputsManual.logits
predictedClass = logitsManual.argmax().item()

labelsMap = {0: "NEGATIVE", 1: "POSITIVE"}
predictedLabelManual = labelsMap[predictedClass]

probabilities = F.softmax(logitsManual, dim=-1)

confidenceScore = probabilities[0, predictedClass].item()

print(f"\nText: {textToAnalyze}")
print(f"Prediksi Sentimen: {predictedLabelManual}")
print(f"Confidence Score: {confidenceScore:.4f}")