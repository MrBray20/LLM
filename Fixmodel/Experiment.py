from mistral import Mistral
from llama import LLAMA
from template import Template
from gemma import Gemma

modelMistral = Mistral()
promptMistral = Template.promptSentimentAnalysis("The conference was well-organized, and the speakers were very knowledgeable.")

print(modelMistral.generateTextPipe(promptMistral))

modelLLAMA = LLAMA()

promptLLAMA = Template.promptSentimentAnalysis("The food at this restaurant was amazing, but the service was a bit slow.")

print(modelLLAMA.generateTextPipe(promptLLAMA))

modelGemma = Gemma()
promptGemma = Template.promptSentimentAnalysis("The food at this restaurant was amazing, but the service was a bit slow.")

print(modelGemma.generateTextPipe(promptGemma))



