import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from deep_translator import GoogleTranslator

load_dotenv()

os.environ['GOOGLE_API_KEY'] = 'AIzaSyAQmgOq7z-n3yCotriI6-W3wpzIDap6Xqg'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

st.title("When Krishna says that....")

@st.cache_resource
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(data)

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")
    faiss_index_path = "krishna_says_faiss_index_multilingual"
    
    if os.path.exists(faiss_index_path):
        return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = load_and_process_pdf("/content/The_Journey_of_Self_Discovery.pdf")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(faiss_index_path)
        return vectorstore

vectorstore = get_vectorstore()

def translate(text, source='auto', target='en'):
    translator = GoogleTranslator(source=source, target=target)
    return translator.translate(text)

def rag_function(query, vectorstore, model, use_gujarati=False):
    relevant_docs = vectorstore.similarity_search(query, k=5)
    
    if use_gujarati:
        context = "\n".join([translate(doc.page_content, source='en', target='gu') for doc in relevant_docs])
    else:
        context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""
    Use the following context to answer the question. If the answer is not in the context then say that wait for the future.. it will show the path by itself,
    response should be psychological and more relevant to spirituality.
    use your knowledge about Krishna's teachings to provide a relevant response.
    give response in very humble way so that the seeker can get relief from their overthinking,
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {query}

    Answer:
    """
    response = model.generate_content(prompt)
    
    return response.text, context

st.sidebar.title("Language Selection")
use_gujarati = st.sidebar.checkbox("Ask in Gujarati")

if use_gujarati:
    query = st.text_input("તમારો પ્રશ્ન ગુજરાતીમાં પૂછો:")
else:
    query = st.text_input("Ask a question about Krishna's teachings:")

if query:
    st.write("Your query:")
    st.markdown(f"**{query}**")

    with st.spinner("Searching for an answer..."):
        model = genai.GenerativeModel('gemini-pro')
        
        if use_gujarati:
            response, context = rag_function(query, vectorstore, model, use_gujarati=True)
            st.subheader("જવાબ:")
            st.markdown(f'<p>{response}</p>', unsafe_allow_html=True)
        else:
            response, context = rag_function(query, vectorstore, model)
            st.subheader("Answer:")
            st.write(response)
    
    with st.expander("View Retrieved Context"):
        st.write(context)

if st.button("Was this answer helpful?"):
    st.write("Always remember - Krushnam sada sahayate! have a good day, jai shree krishna.")
