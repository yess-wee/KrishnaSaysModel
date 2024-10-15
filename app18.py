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

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-image: url('https://wallpapercave.com/wp/wp3037399.jpg');
        background-size: cover;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.8);
    }
    .stMarkdown {
        color: #FFD700;
    }
    .css-1wbqy5l {
        background-color: rgba(25, 25, 112, 0.7);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    sentence_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return sentence_model, tokenizer, model

sentence_model, tokenizer, relevance_model = load_models()

@st.cache_resource
def load_and_process_pdf(uploaded_file):
    with st.spinner("Processing PDF..."):
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("uploaded_pdf.pdf")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(data)

@st.cache_resource
def get_vectorstore():
    faiss_index_path = "krishna_says_faiss_index_multilingual"

    if os.path.exists(faiss_index_path):
        try:
            return FAISS.load_local(faiss_index_path, sentence_model.encode, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading existing index: {e}")

    uploaded_file = st.file_uploader("Upload the PDF file for processing", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing PDF and creating index..."):
            docs = load_and_process_pdf(uploaded_file)
            texts = [doc.page_content for doc in docs]
            embeddings = sentence_model.encode(texts)
            vectorstore = FAISS.from_embeddings(zip(texts, embeddings), sentence_model.encode)
            
            try:
                vectorstore.save_local(faiss_index_path)
                st.success("Index created and saved successfully.")
            except Exception as e:
                st.error(f"Error saving index: {e}")
            
            return vectorstore
    else:
        st.error("Please upload a PDF file to create the index.")
        return None

def translate(text, source='auto', target='en'):
    return GoogleTranslator(source=source, target=target).translate(text)

def llm_extract_relevant_text(query, context, tokenizer, model):
    inputs = tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = model(**inputs)
    relevance_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1]
    
    sentences = context.split('.')
    sentence_scores = []
    for sentence in sentences:
        inputs = tokenizer(query, sentence, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        score = torch.nn.functional.softmax(outputs.logits, dim=1)[0, 1].item()
        sentence_scores.append((sentence, score))
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in sentence_scores[:10]]  # top 10 most relevant sentences
    return ' '.join(top_sentences)

def rag_function(query, vectorstore, llm_model, tokenizer, relevance_model, use_gujarati=False):
    if vectorstore is None:
        return "Error: Vectorstore not initialized. Please upload a PDF.", ""

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    if use_gujarati:
        english_query = translate(query, source='gu', target='en')
    else:
        english_query = query
    
    relevant_text = llm_extract_relevant_text(english_query, context, tokenizer, relevance_model)
  
    prompt = f"""
    Based on Krishna's teachings, answer the following question using the given context.
    If the answer is not in the context, draw from Krishna's wisdom to provide guidance.
    Respond in a humble and spiritual manner, offering solace and enlightenment to the seeker.

    Context: {relevant_text}
    Question: {english_query}
    Krishna's answer:
    """
    response = llm_model.generate_content(prompt).text

    if use_gujarati:
        response = translate(response, source='en', target='gu')
        relevant_text = translate(relevant_text, source='en', target='gu')
    
    return response, relevant_text

# Main Streamlit app
st.title("Krishna Says")

vectorstore = get_vectorstore()

st.sidebar.title("Language Selection")
use_gujarati = st.sidebar.checkbox("Ask in Gujarati")

query = st.text_input("Ask Krishna for guidance:" if not use_gujarati else "તમારો પ્રશ્ન ગુજરાતીમાં પૂછો:")

if query:
    st.write("Your question to Krishna:")
    st.markdown(f"**{query}**")
    
    with st.spinner("Krishna is contemplating..."):
        llm_model = genai.GenerativeModel('gemini-pro')
        response, context = rag_function(query, vectorstore, llm_model, tokenizer, relevance_model, use_gujarati)
    
    st.subheader("Krishna says:")
    st.markdown(f'<p style="color: #FFD700; font-style: italic;">{response}</p>', unsafe_allow_html=True)
    
    with st.expander("View Relevant Teachings"):
        st.write(context)

if st.button("Was Krishna's wisdom helpful?"):
    st.write("🕉️ Always remember... Krushnam sadasahayate..🕉️")
