from langchain_community.document_loaders import WebBaseLoader
import os

url = "https://docs.langchain.com/oss/python/integrations/document_loaders/index#all-document-loadersl"

data = WebBaseLoader(url)
doc = data.load()

print(doc[0].page_content)

