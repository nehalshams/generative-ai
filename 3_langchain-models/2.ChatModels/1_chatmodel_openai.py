from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI(model = 'gpt-5.5')

result1 = model1.invoke("What is the capital of india")

print(result1, result1.content)


model2 = ChatOpenAI(
    model = 'gpt-5.5', 
    temperature = 1.5, 
    # max_completion_token = 100
    )

result2 = model2.invoke("Write five lines of poem on cricket")
print(result2.content)







