from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

import langchain_core
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0
)

template  = ChatPromptTemplate.from_messages([
     ("system", "You are Ai that summarize the text"),
        ("human", "{data}")
])


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "ASME_ARTICAL_16.pdf")

data = PyPDFLoader(file_path)

doc = data.load()

final_prompt = template.format_messages(data = doc[0].page_content)

response = model.invoke(final_prompt)

print(response.content)