from langchain_community.document_loaders import TextLoader
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "note.txt")
data = TextLoader(file_path)

docs = data.load()

print(docs[0].page_content)