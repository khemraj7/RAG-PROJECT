# laod Pdf
# Split into Chunks
# Create Embeddings
# save to vector db
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF document
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "deep_learning.pdf")


data = PyPDFLoader(file_path)
docs = data.load()

# Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

print(f"Number of chunks created: {len(chunks)}")

# Create embeddings and save to vector database
embedding_model = MistralAIEmbeddings()


# Create a Chroma vector store and persist the embeddings
vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Documents added to the vector store successfully.")


