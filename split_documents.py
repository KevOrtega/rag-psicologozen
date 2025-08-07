from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from language_detector import detect_pdf_language
from config import DOCUMENT_PATHS

def split_documents(chunk_size=1000, chunk_overlap=200):
    documents = []
    for path in DOCUMENT_PATHS:
        if path.endswith(".pdf"):
            detected_lang = detect_pdf_language(path)
            loader = UnstructuredPDFLoader(
                path,
                mode="elements",
                languages=[detected_lang]
            )
        elif path.endswith(".docx" or ".doc"):
            loader = UnstructuredWordDocumentLoader(path,mode="elements")
        else:
            raise ValueError(f"Unsupported file type: {path}")
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
