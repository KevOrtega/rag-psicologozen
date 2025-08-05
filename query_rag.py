from llm_chain import create_rag_chain
from embedder import load_vectorstore

def query_rag_chain():
    print("🧠 Cargando base vectorial de Qdrant...")
    vectorstore = load_vectorstore()
    rag_chain = create_rag_chain(vectorstore)

    query = input("🔎 Ingrese su consulta clínica: ")
    respuesta = rag_chain.invoke(query)
    print("\n📋 Respuesta:")
    print(respuesta['result'])

if __name__ == "__main__":
    query_rag_chain()
