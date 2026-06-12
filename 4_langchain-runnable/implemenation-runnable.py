from abc import ABC, abstractmethod
import random

class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_dict):
        pass

class NakliLLM(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list = [
            "Delhi is the capital of India.",
            "IPL is a popular cricket league in India.",
            "Python is a widely used programming language.",
            "The Taj Mahal is a famous monument in India.",
        ]

        return { 'response': random.choice(response_list) }

    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India.",
            "IPL is a popular cricket league in India.",
            "Python is a widely used programming language.",
            "The Taj Mahal is a famous monument in India.",
        ]

        return { 'response': random.choice(response_list) }
    


class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        print("Prompt Template created")

    def invoke(self, input_dict):
        return self.format(input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)
    
class RunnableConnector(Runnable):
    def __init__(self, runnanle_list):
        self.runnable_list = runnanle_list
        print("Runnable Connector created")

    def invoke(self, input_dict):
        result = input_dict
        for runnable in self.runnable_list:
            result = runnable.invoke(result)
        return result