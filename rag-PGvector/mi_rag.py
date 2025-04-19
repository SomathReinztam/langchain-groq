"""
mi_rag.py

"""

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
text = r"C:\Users\Acer\Documents\python\OpenScienceAI\2_chatbot\test.txt"


raw_documents = TextLoader(text, encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

embeddings_model = HuggingFaceEmbeddings()

db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

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
