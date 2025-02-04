from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env file!")

app = FastAPI()

# Enable CORS for all origins
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
# Function to Load PDF and Create Vector Store
def get_vectorstore_from_static_pdf(pdf_path=PDF_PATH):
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None values

        # Split text into chunks
        from langchain.text_splitter import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)

        # Create a vectorstore from the chunks
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)

        return vector_store
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

# Load the vector store at startup
vector_store = get_vectorstore_from_static_pdf()

# Chat request model
class ChatRequest(BaseModel):
    message: str

# Function to get response from OpenAI
def get_response(user_input):
    if not vector_store:
        return "Vector store is not initialized. Check the PDF path."
    
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever()
    
    response = llm.invoke(user_input)  # Directly using LLM for response
    return response.content

# API Endpoint for Chat
@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    response = get_response(chat_request.message)
    return {"answer": response}
