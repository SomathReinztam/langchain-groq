"""
mi_rag.py

"""

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore  # aqui
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
import pinecone  # aqui

# Configuración de Pinecone (reemplaza con tus credenciales)
PINECONE_API_KEY = "tu-api-key-de-pinecone"
PINECONE_ENVIRONMENT = "tu-environment"  # ej: "gcp-starter"
PINECONE_INDEX_NAME = "nombre-de-tu-indice"  # créalo primero en pinecone.io

text = r"C:\Users\Acer\Documents\python\OpenScienceAI\2_chatbot\test.txt"

# Cargar y dividir documentos (igual que antes)
raw_documents = TextLoader(text, encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Modelo de embeddings (igual que antes)
embeddings_model = HuggingFaceEmbeddings()

# Inicializar Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Crear o cargar el índice en Pinecone (reemplaza PGVector)
db = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings_model,
    index_name=PINECONE_INDEX_NAME
)

retriever = db.as_retriever(search_kwargs={"k": 2})

query = 'Who are the key figures in the ancient greek history of philosophy?'


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)



GROQ_API_KEY = "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"
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=1
)


@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer


result = qa.invoke(query)
print(result.content)
