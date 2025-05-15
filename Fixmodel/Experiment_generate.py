from mistral import Mistral
from llama import LLAMA
from template import Template
from gemma import Gemma
import pandas as pd
import re
import json

modelMistral = Mistral()

modelLLAMA = LLAMA()

modelGemma = Gemma()

# prompt = Template.promptEmail()

# parameterEmail = {"temperature": 0.7,
#         "do_sample": True,
#         "max_new_tokens":200}
# resultMistral = modelMistral.generateTextPipeArgs(prompt,parameterEmail)
# resultLLAMA = modelLLAMA.generateTextPipeArgs(prompt,parameterEmail)
# resultGemma = modelGemma.generateTextPipeArgs(prompt,parameterEmail)
# print("-------------------------------mistral------------------------------------")
# print(resultMistral)
# print("-------------------------------LLAMA--------------------------------------")
# print(resultLLAMA)
# print("-------------------------------Gemma--------------------------------------")
# print(resultGemma)


prompt = Template.promptSocialMedia()

parameterEmail = {"temperature": 0.8,
        "do_sample": True}

resultMistral = modelMistral.generateTextPipeArgs(prompt,parameterEmail)
resultLLAMA = modelLLAMA.generateTextPipeArgs(prompt,parameterEmail)
resultGemma = modelGemma.generateTextPipeArgs(prompt,parameterEmail)
print("-------------------------------mistral------------------------------------")
print(resultMistral)
print("-------------------------------LLAMA--------------------------------------")
print(resultLLAMA)
print("-------------------------------Gemma--------------------------------------")
print(resultGemma)