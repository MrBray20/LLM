# from llm.modelManager import ModelManager
# from llm.template import Template
class Runner:
    def __init__(self):
        ...
        # self.manager = ModelManager()
        
    def run_model_task(self, model_name, **kwargs):
        try:
            result = self.manager.run_task(model_name, **kwargs)
            return result
        except Exception as e:
            return f"Error: {str(e)}"
        
    def run_all(self,on_result,**kwargs):
        # print(kwargs)
        ...
        # task = kwargs["task"]
        # if task == "Text Generation":
            # try:
                # return self.manager.run_all_model(on_result,**kwargs)
            # except Exception as e:
                # return f"Error: {str(e)}"
        # elif task == "Sentiment Analysis":
            # text= Template.promptSentimentAnalysis(kwargs["prompt"])
            # kwargs["prompt"]=text
            # print(kwargs)
            # try:
                # return self.manager.run_all_model_sentimen(on_result,**kwargs)
            # except Exception as e:
                # return f"Error: {str(e)}"
        


    
