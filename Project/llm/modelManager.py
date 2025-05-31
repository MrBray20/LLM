import json
import re
from llm.mistral import Mistral
from llm.llama import LLaMA
from llm.gemma import Gemma 
from configparser import ConfigParser

config = ConfigParser()
config.read("./config.ini")


class ModelManager:
    def __init__(self):
        self.model_paths = {
            "mistral": config["Paths"]["Mistral"],
            "llama" : config["Paths"]["LLaMA"],
            "gemma" : config["Paths"]["Gemma"]
        }
        self.__models = {}
        
        # self.__models = {
        #     "mistral": 1,
        #     "llama" : 2,
        #     "gemma" : 3
        # }
    # def run_task(self,model_name,**kwargs):
    #     if model_name not in self.__models:
    #         raise ValueError(f"Model '{model_name}' tidak ditemukan")
    #     model = self.__models[model_name]
    #     if "prompt" in kwargs:
    #         result = model.generateTextPipe(kwargs["prompt"])
    #     return result
    def _load_model(self, model_name):
        if model_name not in self.__models:
            if model_name == "mistral":
                self.__models[model_name] = Mistral(model_name=self.model_paths[model_name])
            elif model_name == "llama":
                self.__models[model_name] = LLaMA(model_name=self.model_paths[model_name])
            elif model_name == "gemma":
                self.__models[model_name] = Gemma(model_name=self.model_paths[model_name])
            else:
                raise ValueError(f"Model '{model_name}' tidak dikenal")
    
    def run_all_model(self, on_result=None, **kwargs):
        results = {}
        for model_name in kwargs["models"]:
            print(f"Menjalankan model: {model_name}")
            try:
                self._load_model(model_name)
                result = self.__models[model_name].generateTextPipe(kwargs["prompt"])
                results[model_name] = result
                if on_result is not None:
                    on_result(model_name, result)
            except Exception as e:
                print(f"Model {model_name} gagal dijalankan: {e}")
        return results
    
    def run_all_model_sentimen(self, on_result=None, **kwargs):
        def process_model(model, prompt, pattern, model_name):
            while True:
                try:
                    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {model_name} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    result = model.generateTextPipeSentimen(prompt)
                    # print(result)
                    json_result = json.loads(re.search(pattern, result).group())
                    # print(f"{model_name}:", json_result)
                    return json_result
                except Exception as e:
                    print(f"[ERROR] {model_name} gagal memproses")
                    print(f"Mengulang kembali proses")
                    
        json_pattern=r'\{[^{}]*\}'
        results = {}
        for model_name in kwargs["models"]:
            print(f"Menjalankan model: {model_name}")
            try:
                self._load_model(model_name)
                jsontextMistral=process_model(self.__models[model_name],kwargs["prompt"],json_pattern,model_name)
                # result = self.__models[model_name].generateTextPipeSentimen(kwargs["prompt"])
                # results[model_name] = result
                print(jsontextMistral)
                if on_result is not None:
                    on_result(model_name, jsontextMistral)
            except Exception as e:
                print(f"Model {model_name} gagal dijalankan: {e}")
        return results

    
    # def run_all_model(self, on_result=None, **kwargs):
    #     results={}
    #     # print(kwargs)
    #     for model_name in kwargs["models"]:
    #         print(f"Menjalankan model: {model_name}")
    #         try:
    #             result = self.__models[model_name].generateTextPipe(kwargs["prompt"])
                
    #             # result = f"test {model_name}"
    #             results[model_name] = result
                
    #             if on_result is not None:
    #                 on_result(model_name, result)

    #         except Exception as e:
    #             print(f"Model {model_name} gagal dijalankan: {e}")
            
    #     return results