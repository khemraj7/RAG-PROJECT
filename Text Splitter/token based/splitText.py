from langchain_text_splitters import TokenTextSplitter
import os
    
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "state_of_the_union.txt")

# Load example document
with open(file_path) as f:
    docs = f.read()
    
text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

texts = text_splitter.split_text(docs)

for i, text in enumerate(texts):
    print(f"Chunk {i + 1}:")
    print(text)
    print()