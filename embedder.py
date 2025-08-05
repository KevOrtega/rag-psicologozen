from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, QDRANT_ENDPOINT

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = QdrantClient(url=QDRANT_ENDPOINT)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="clinical_docs",
        embedding=embeddings,
    )
    return vectorstore

def index_documents(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = QdrantClient(url=QDRANT_ENDPOINT)
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client,
        collection_name="clinical_docs",
    )
