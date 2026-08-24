from dotenv import load_dotenv
import os
load_dotenv()

import os
print("MISTRAL_API_KEY:", os.getenv("MISTRAL_API_KEY"))

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader

model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0
)


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "note.txt")
data = TextLoader(file_path)

template = ChatPromptTemplate.from_messages([
     ("system", "You are Ai that summarize the text"),
     ("human", "{data}")
])

docs = data.load()

final_prompt = template.format_messages(data = docs[0].page_content)

response = model.invoke(final_prompt)

print(response.content)