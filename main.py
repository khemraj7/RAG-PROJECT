from dotenv import load_dotenv

load_dotenv()
import os
print("MISTRAL_API_KEY:", os.getenv("MISTRAL_API_KEY"))

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0
)

response = model.invoke("hello ?")

print(response.content)