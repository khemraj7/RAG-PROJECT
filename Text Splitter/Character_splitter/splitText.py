import os
from langchain_text_splitters import CharacterTextSplitter



script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "note.txt")

# Load example document
with open(file_path) as f:
    docs = f.read()
    
    
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(docs)    

for i, text in enumerate(texts):
    print(f"Chunk {i + 1}:")
    print(text)
    print()
    