import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from deep_translator import GoogleTranslator

# Load environment variables and configure Google API
load_dotenv()
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAQmgOq7z-n3yCotriI6-W3wpzIDap6Xqg'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit page configuration
st.set_page_config(page_title="Krishna Says", page_icon="🕉️", layout="wide")

# Custom CSS with animations and black background
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: #FFD700;
    }
    .stMarkdown {
        color: #FFD700;
    }
    .css-1wbqy5l {
        background-color: rgba(25, 25, 112, 0.7);
    }
    @keyframes flute-animation {
        0% { transform: rotate(0deg); }
        50% { transform: rotate(5deg); }
        100% { transform: rotate(0deg); }
    }
    .flute-icon {
        animation: flute-animation 3s infinite;
        display: inline-block;
    }
    @keyframes peacock-feather-animation {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .peacock-feather {
        animation: peacock-feather-animation 4s infinite;
        display: inline-block;
    }
    .st-emotion-cache-10trblm {
        position: relative;
        overflow: hidden;
    }
    .st-emotion-cache-10trblm::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,215,0,0.2), transparent);
        animation: shine 3s infinite;
    }
    @keyframes shine {
        100% { left: 100%; }
    }
    .stButton > button {
        background-color: #4B0082;
        color: #FFD700;
        border: 2px solid #FFD700;
    }
    .stButton > button:hover {
        background-color: #800080;
    }
</style>
""", unsafe_allow_html=True)

# Rest of your code remains the same...

# Main Streamlit app
st.title("🕉️ Krishna Says 🕉️")
st.markdown("<div class='flute-icon'>🎶</div> <div class='peacock-feather'>🦚</div>", unsafe_allow_html=True)

vectorstore = get_vectorstore()

st.sidebar.title("Language Selection")
use_gujarati = st.sidebar.checkbox("Ask in Gujarati")

query = st.text_input("Ask Krishna for guidance:" if not use_gujarati else "તમારો પ્રશ્ન ગુજરાતીમાં પૂછો:")

if query:
    st.write("Your question to Krishna:")
    st.markdown(f"**{query}**")

    with st.spinner("Krishna is contemplating... 🧘"):
        llm_model = genai.GenerativeModel('gemini-pro')
        response, context = rag_function(query, vectorstore, llm_model, tokenizer, relevance_model, use_gujarati)

    st.subheader("Krishna says:")
    st.markdown(f'<p style="color: #FFD700; font-style: italic; text-shadow: 0 0 5px #FFA500;">{response}</p>', unsafe_allow_html=True)

    with st.expander("View Relevant Teachings"):
        st.write(context)

if st.button("Was Krishna's wisdom helpful? 🙏"):
    st.balloons()
    st.write("🕉️ Always remember... Krushnam sadasahayate..🕉️")
