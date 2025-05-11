from mistral import Mistral
from llama import LLAMA
from template import Template
from gemma import Gemma
import pandas as pd
import re
import json


modelMistral = Mistral()
# promptMistral = Template.promptStory("adventure","tarzan from the jugler")
# promptMistral = Template.promptBebas("Terjemahkan kalimat berikut ke dalam bahasa Inggris: 'Saya lapar.'")

# print(modelMistral.generateTextPipe(promptMistral))

modelLLAMA = LLAMA()

# promptLLAMA = Template.promptSentimentAnalysis("The food at this restaurant was amazing, but the service was a bit slow.")
# promptLLAMA = Template.promptStory("adventure","tarzan from the jugler")

# print(modelLLAMA.generateTextPipe(promptLLAMA))

modelGemma = Gemma()
# # promptGemma = Template.promptSentimentAnalysis("The food at this restaurant was amazing, but the service was a bit slow.")
# promptGemma = Template.promptStory("adventure","tarzan from the jugler")

# print(modelGemma.generateTextPipe(promptGemma))


df = pd.read_csv(r"D:\SKRIPSI\Code Program\Fixmodel\archive\test.csv", encoding="latin1")
df = df.dropna(subset=['text'])
dftest = df[["text"]]
dfactual = df[["sentiment"]]

start_index = None

try:
    with open("last_index.txt", "r") as f:
        start_index = int(f.read()) + 1
except FileNotFoundError:
    start_index = 0

resultPredicMistral=[]
resultpredicLLAMA=[]
resultpredicGemma=[]
text_indices = []

json_pattern=r'\{[^{}]*\}'

try:
    for index ,row in dftest.iloc[(start_index or 0):].iterrows():
        print(f"============================== {index} ======================================")
        teks = row['text']
        promptMistral = Template.promptSentimentAnalysis(teks)
        resultMistral = modelMistral.generateTextPipe(promptMistral)
        print(resultMistral)
        jsontextMistral = json.loads(re.search(json_pattern,resultMistral).group())
        resultPredicMistral.append(jsontextMistral)
        print("Mistral")
        print(jsontextMistral)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        text_indices.append(index)
        promptLLAMA = Template.promptSentimentAnalysis(teks)
        resultLLAMA = modelLLAMA.generateTextPipe(promptLLAMA)
        print(resultLLAMA)
        jsontextLLAMA = json.loads(re.search(json_pattern,resultLLAMA).group())
        resultpredicLLAMA.append(jsontextLLAMA)
        print("LLAMA")
        print(jsontextLLAMA)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        promptGemma = Template.promptSentimentAnalysis(teks)
        resultGEmma = modelGemma.generateTextPipe(promptGemma)
        print(resultGEmma)
        jsontextGemma = json.loads(re.search(json_pattern,resultGEmma).group())
        resultpredicGemma.append(jsontextGemma)
        print("Gemma")
        print(jsontextGemma)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Saving current result...")
finally:
    min_len = min(len(text_indices), len(resultPredicMistral), len(resultpredicLLAMA), len(resultpredicGemma))
    text_indices = text_indices[:min_len]
    resultPredicMistral = resultPredicMistral[:min_len]
    resultpredicLLAMA = resultpredicLLAMA[:min_len]
    resultpredicGemma = resultpredicGemma[:min_len]
    
    dfpredicModel = pd.DataFrame({
        'index':text_indices,
        'PredicMistral':resultPredicMistral,
        'PredicLLAMA':resultpredicLLAMA,
        'PredicGemma':resultpredicGemma
        }).set_index('index')
    
    
    file_path = "resultPredicMode2-1.csv"
    write_header = not pd.io.common.file_exists(file_path)
    
    dfpredicModel.to_csv(file_path, mode='a', header=write_header)
    print(f"Hasil disimpan ke {file_path}")
    # print(dfpredicModel)
    # dfpredicModel.to_pickle("resultPredicModel.pkl")
    # print("Results saved to resultPredicModel.pkl")
    
    if text_indices:
        last_index = text_indices[-1]
        with open("last_index.txt", "w") as f:
            f.write(str(last_index))
        print(f"Last index saved to last_index.txt: {last_index}")