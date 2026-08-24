from langchain_community.document_loaders import PyPDFLoader
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "ASME_ARTICAL_16.pdf")

data = PyPDFLoader(file_path)

doc = data.load()

print(doc[0].page_content)