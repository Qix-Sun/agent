# RAG system setup using LangChain with Qwen3 and Qwen3-Embedding-4B
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.document_loaders import TextLoader
import os

# ============ Embedding Model via Ollama ============
embedding_model = OllamaEmbeddings(
    model="dengcao/Qwen3-Embedding-4B:Q4_K_M",
    base_url=""
)

# Load existing FAISS vector DB
vectorstore = FAISS.load_local(
    folder_path="vector_db",
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True
)

# ============ LLM Backend (Qwen3-8B via Ollama) ============
llm = ChatOllama(
    model="qwen3:8b",
    base_url="",
    temperature=0.7,
    top_p=0.95
)

# ============ RAG Chain Setup ============
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ============ Simple Query Function for Testing ============
def ask_ophthalmology_agent(query: str):
    result = rag_chain(query)
    return result['result'], result['source_documents']

# Example test
if __name__ == "__main__":
    answer, sources = ask_ophthalmology_agent("什么是白内障？")
    print("Answer:\n", answer)
    print("\nSources:\n", sources)
