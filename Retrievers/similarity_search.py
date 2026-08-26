from dotenv import load_dotenv
load_dotenv()


from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

# Create embeddings and save to vector database
embedding_model = MistralAIEmbeddings()

# chroma vector store
vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n===== Similarity Search Results =====\n")

similarity_docs = vectorstore.similarity_search("GPUs, TPUs, and batches?", k=3)

for doc in similarity_docs:
    print(doc.page_content)