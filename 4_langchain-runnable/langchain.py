import random

class NakliLLM:
    def __init__(self):
        print("LLM created")

    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India.",
            "IPL is a popular cricket league in India.",
            "Python is a widely used programming language.",
            "The Taj Mahal is a famous monument in India.",
        ]

        return { 'response': random.choice(response_list) }
    


llm = NakliLLM()

print(llm.predict("What is the capital of India?"))


class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        print("Prompt Template created")

    def format(self, input_dict):
        return self.template.format(**input_dict)
    

template = NakliPromptTemplate("Write a {length} poem about {topic}?", ["length", "topic"])
result = template.format({"length": "short", "topic": "India"})
print(result)


# use prompt template result as input to llm
print(llm.predict(result))



# Create a chain that combines the prompt template and the LLM
class NakliChain:
    def __init__(self, prompt_template, llm):
        self.prompt_template = prompt_template
        self.llm = llm
        print("Chain created")

    def run(self, input_dict):
        formatted_prompt = self.prompt_template.format(input_dict)
        return self.llm.predict(formatted_prompt)

# Example usage
chain = NakliChain(template, llm)
print(chain.run({"length": "short", "topic": "India"}))