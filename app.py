from fastapi import FastAPI
from pydantic import BaseModel
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("ERROR: OPENAI_API_KEY is missing from .env file!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded PDF path
# PDF_PATH = "C:/Users/lenovo/Downloads/ApexDeveloperGuidea.pdf"
PDF_PATH = "/home/ubuntu/ApexDeveloperGuidea.pdf"
# Function to get vector store from PDF
def get_vectorstore_from_static_pdf(pdf_path=PDF_PATH):
    pdf_reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# **✅ FIX: Use ChatOpenAI() instead of OpenAIEmbeddings()**
def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)  # ✅ Use LLM, not Embeddings

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to get relevant information.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# Function to get conversational RAG chain
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0) 

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions only based on the below context:\n\n{context}\n\n"
                   "If the context does not contain relevant information, respond with 'I don't know'."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Function to get response
# def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    retrieved_docs = retriever_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    if not retrieved_docs['documents']:
        return "I don't know. The PDF does not contain relevant information."

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    retrieved_docs = retriever_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    if not retrieved_docs:  
        return "I don't know. The PDF does not contain relevant information."

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

# Streamlit App
st.set_page_config(page_title="Chat with PDF")
st.title("Chat with PDF")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

if "vector_store" not in st.session_state:
    with st.spinner("Processing PDF..."):
        st.session_state.vector_store = get_vectorstore_from_static_pdf()

user_query = st.chat_input("Type your message here...")
if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)
