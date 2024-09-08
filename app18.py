import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

load_dotenv()


os.environ['GOOGLE_API_KEY'] = 'AIzaSyAQmgOq7z-n3yCotriI6-W3wpzIDap6Xqg'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = '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'
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


st.title("When Krishna says that....")

loader = PyPDFLoader("/content/bhagwadgeeta.pdf")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)


vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})


llm = ChatGoogleGenerativeAI(
    api_key=os.environ['GOOGLE_API_KEY'],
    api_base="https://api.groq.com/genai/v1",
    model="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)


rag_tool = PDFSearchTool(
    pdf='/content/bhagwadgeeta.pdf',
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


query = st.chat_input("Ask Anything: ") 

if query:

    response = rag_tool.run(query)
    st.write(response)
