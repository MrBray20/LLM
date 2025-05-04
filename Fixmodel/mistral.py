from ModelStruktur import ModelStruktur
class Mistral(ModelStruktur):
    def __init__(self,model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"):
        super().__init__(model_name)