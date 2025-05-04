from mistral import Mistral
from llama import LLAMA
from template import Template
from gemma import Gemma

modelMistral = Mistral()
# promptMistral = Template.promptSentimentAnalysis("The conference was well-organized, and the speakers were very knowledgeable.")
promptMistral = Template.promptStory("adventure","tarzan from the jugler")

print(modelMistral.generateTextPipe(promptMistral))

modelLLAMA = LLAMA()

# promptLLAMA = Template.promptSentimentAnalysis("The food at this restaurant was amazing, but the service was a bit slow.")
promptLLAMA = Template.promptStory("adventure","tarzan from the jugler")

print(modelLLAMA.generateTextPipe(promptLLAMA))

modelGemma = Gemma()
# promptGemma = Template.promptSentimentAnalysis("The food at this restaurant was amazing, but the service was a bit slow.")
promptGemma = Template.promptStory("adventure","tarzan from the jugler")

print(modelGemma.generateTextPipe(promptGemma))







