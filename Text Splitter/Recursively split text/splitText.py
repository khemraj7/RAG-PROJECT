from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "note.txt")

# Load example document
with open(file_path) as f:
    docs = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([docs])
# print(texts)

for i, text in enumerate(texts):
    print(f"Chunk {i + 1}:")
    print(text.page_content)
    print()
