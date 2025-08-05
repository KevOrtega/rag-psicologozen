from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from language_detector import detect_pdf_language
from config import PDF_PATHS

def load_and_split_documents(chunk_size=1000, chunk_overlap=200):
    documents = []
    for path in PDF_PATHS:
        detected_lang = detect_pdf_language(path)
        loader = UnstructuredPDFLoader(
            path,
            mode="elements",
            languages=[detected_lang]
        )
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
