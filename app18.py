import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import faiss
from langchain.vectorstores import FAISS
import os

load_dotenv()

os.environ['GOOGLE_API_KEY'] = 'AIzaSyAQmgOq7z-n3yCotriI6-W3wpzIDap6Xqg'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = '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'
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

st.title("When Krishna says that....")

# loader = PyPDFLoader("/content/The_Journey_of_Self_Discovery.pdf")
loader = PyPDFLoader("The_Journey_of_Self_Discovery.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


faiss_index_path = r"D:\KrishnaSaysDB\faiss_index"
print(os.listdir(directory))

if os.path.exists(faiss_index_path):

    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    #st.write("Loaded FAISS index from:", faiss_index_path)
else:
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    # Save the FAISS index to a local file
    vectorstore.save_local(faiss_index_path)
    #st.write("FAISS index created and saved to:", faiss_index_path)

# List the contents of the directory to verify the FAISS index
# st.write("Directory contents:")
# st.write(os.listdir("/content"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(
    api_key=os.environ['GOOGLE_API_KEY'],
    api_base="https://api.groq.com/genai/v1",
    model="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)

rag_tool = PDFSearchTool(
    pdf='The_Journey_of_Self_Discovery.pdf',
    config=dict(
        llm=dict(
            provider="groq",  
            config=dict(
                model="llama3-8b-8192",
            ),
        ),
        embedder=dict(
            provider="huggingface",  
            config=dict(
                model="BAAI/bge-small-en-v1.5",
            ),
        ),
    )
)

query = st.chat_input("Say something: ")

if query:
    response = rag_tool.run(query)
    st.write(response)
