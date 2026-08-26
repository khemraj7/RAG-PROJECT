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

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("\n===== MMR Results =====\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)