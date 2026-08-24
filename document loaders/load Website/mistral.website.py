from langchain_community.document_loaders import WebBaseLoader
import os

from dotenv import load_dotenv
load_dotenv()

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


url = "https://docs.langchain.com/oss/python/integrations/document_loaders/index#all-document-loadersl"

data = WebBaseLoader(url)
doc = data.load()

final_prompt = template.format_messages(data = doc[0].page_content)

response = model.invoke(final_prompt)

print("-----------------------------This content is summarized by the AI model Mistral : ------------------------------------------")
print(response.content)

# invertedCommas = '"'