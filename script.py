from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # NUEVO import correcto

# 1. Cargar PDFs
pdf_paths = ["document1.pdf", "document2.pdf"]
loaders = [PyPDFLoader(path) for path in pdf_paths]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# 2. Dividir texto
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)

# 3. Embeddings con BioBERT vía HuggingFace
model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 4. Crear base FAISS
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 5. Usar OpenRouter como proveedor LLM
llm = ChatOpenAI(
    openai_api_key="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",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/horizon-beta"
)

# 6. Cadena RAG
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 7. Consulta
query = "¿Qué diagnósticos clínicos aparecen en los documentos?"
respuesta = rag_chain.invoke(query)

print("Respuesta del modelo vía OpenRouter:")
print(respuesta['result'])
