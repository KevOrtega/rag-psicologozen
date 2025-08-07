from split_documents import split_documents
from embedder import index_documents

def save_documents():
    # 1. Cargar y dividir documentos
    print("Cargando y procesando documentos...")
    docs = split_documents()

    # 2. Guardar documentos
    print("Creando embeddings...")
    index_documents(docs)
    print("Documentos guardados")

if __name__ == "__main__":
    save_documents()
