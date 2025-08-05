from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


def create_rag_chain(vectorstore):
    llm = ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        model=LLM_MODEL
    )

    prompt_template = """
    Eres un asistente que busca información precisa en los documentos proporcionados.
    No hagas interpretaciones ni diagnósticos, solo responde exactamente con la información solicitada.

    Documentos: {context}
    Pregunta: {question}
    Respuesta:
    """
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return rag_chain
