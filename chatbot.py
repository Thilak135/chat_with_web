import streamlit as st
import time

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyTXTLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma, faiss, qdrant

# Define functions for loading components
def create_vector_db(data_path="./example.txt", db_chroma_path="vectorstores/db_chroma"):
    """Creates a vector database from PDF documents in the specified path.

    Args:
        data_path (str, optional): Path to the directory containing PDFs. Defaults to "data/".
        db_chroma_path (str, optional): Path to persist the vector database. Defaults to "vectorstores/db_chroma".
    """
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

    db = Chroma.from_documents(texts, embeddings, persist_directory=db_chroma_path)
    return db

def set_custom_prompt():
    """Returns a custom prompt template for question-answering."""
    prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def retrieval_qa_chain(llm, prompt, db):
    """Creates a retrieval-based QA chain.

    Args:
        llm: The loaded language model.
        prompt: The custom prompt template.
        db: The vector database containing document embeddings.

    Returns:
        RetrievalQA: The configured retrieval-based QA chain.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def load_llm(model_path="C:\\Users\\vidya\\OneDrive\\Desktop\\startup\\llama-2-7b-chat.ggmlv3.q8_0.bin"):
    """Loads the language model.

    Args:
        model_path (str, optional): Path to the language model file. Defaults to "C:\\Users\\vidya\\OneDrive\\Desktop\\startup\\llama-2-7b-chat.ggmlv3.q8_0.bin".

    Returns:
        CTransformers: The loaded language model.
    """
    return CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
    )


# Initialize components (consider loading from configuration file for flexibility)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

# Create the vector database (uncomment and adjust paths if needed)
# db = create_vector_db()

llm = load_llm()
qa_prompt = set_custom_prompt()

try:
    # Attempt to load existing vector database
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
except FileNotFoundError:
    # Handle case where vector database doesn't exist (e.g., prompt user to create it)
    st.error(
        "Vector database not found. Please create one using the `create_vector_db` function or provide the path to an existing database."
    )
    db = None
