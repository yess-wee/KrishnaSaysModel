import re
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
import random

load_dotenv()
# os.environ['GOOGLE_API_KEY'] = 'AIzaSyAQmgOq7z-n3yCotriI6-W3wpzIDap6Xqg'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAV1opSnseVjUsxmCa2QStiuwgFTVktqdc'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

quotes = [
    "You are what you believe in. You become that which you believe you can become.",
    "You are only entitled to the action, never to its fruits.",
    "Death is as sure for that which is born, as birth is for that which is dead. Therefore grieve not for what is inevitable.",
    "One who sees inaction in action, and action in inaction, is intelligent among men.",
    "Arise, slay thy enemies, enjoy a prosperous kingdom.",
    "It is better to live your own destiny imperfectly than to live an imitation of somebody else's life with perfection.",
    "Through selfless service, you will always be fruitful and find the fulfillment of your desires.",
    "Remember, I am always with you, don't be afraid :)",
    "Hare Krishna Hare Krishna, Krishna Krishna Hare Hare, Hare Raam Hare Raam, Raam Raam Hare Hare.",
    "Jai Shree Krishna"
]

st.set_page_config(page_title="Krishna Says", page_icon="🕉️", layout="wide")


st.markdown("""
<style>
    .stApp {
        background-image: url('https://wallpapercave.com/wp/wp3037399.jpg');
        background-size: cover;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        color: black;
    }
    .stMarkdown, .stTitle, .stHeader, .stSubheader, .stText, .stExpanderHeader, .stCaption {
        color: orange;  
    }
    .css-1wbqy5l {
        background-color: rgba(25, 25, 112, 0.7);
    }
    .stButton {
        background-color: rgba(255, 215, 0, 8);
        border: none;
        border-radius: 10px;
    }
    .stExpanderHeader {
        font-weight: bold;
        color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

def display_footer():
    random_quote = random.choice(quotes)
    st.markdown(
        f"""<div style="position: fixed; bottom: 0; width: 100%; background-color: rgba(0, 0, 0, 0.7); color: orange;
             text-align: center; padding: 10px; font-style: italic;">
             "{random_quote}"
        </div>""",
        unsafe_allow_html=True
    )

@st.cache_resource
def load_models():
    try:
        sentence_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        return sentence_model, tokenizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")

sentence_model, tokenizer, relevance_model = load_models()

@st.cache_resource
def load_and_process_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(data)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

@st.cache_resource
def get_vectorstore():
    faiss_index_path = "krishna_says_faiss_index_multilingual"
    pdf_path = "https://raw.githubusercontent.com/yess-wee/KrishnaSaysModel/main/translationofbg.pdf"

    if os.path.exists(faiss_index_path):
        try:
            return FAISS.load_local(faiss_index_path, sentence_model.encode, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading existing index: {e}")

    with st.spinner("Processing pre-uploaded PDF and creating index..."):
        docs = load_and_process_pdf(pdf_path)
        if not docs:
            return None
       
        texts = [doc.page_content for doc in docs]
        embeddings = sentence_model.encode(texts)
        vectorstore = FAISS.from_embeddings(zip(texts, embeddings), sentence_model.encode)

        try:
            vectorstore.save_local(faiss_index_path)
        except Exception as e:
            st.error(f"Error saving index: {e}")
       
        return vectorstore

def translate(text, source='auto', target='en'):
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text


def llm_extract_relevant_text(query, context, tokenizer, model):
    try:
        inputs = tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        relevance_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1]

        # Split sentences more carefully and clean up any repeated numbers or artifacts
        sentences = re.split(r'(?<!\d)\. |\n', context)  # Splits by period or new line without splitting decimal numbers
        sentence_scores = []
        
        for sentence in sentences:
            # Remove leading numbers or extra whitespace
            clean_sentence = re.sub(r'^\d+\s*', '', sentence).strip()
            if not clean_sentence:
                continue  # Skip empty sentences

            inputs = tokenizer(query, clean_sentence, return_tensors="pt", truncation=True, max_length=512, padding=True)
            outputs = model(**inputs)
            score = torch.nn.functional.softmax(outputs.logits, dim=1)[0, 1].item()
            sentence_scores.append((clean_sentence, score))

        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:8]]  
        return ' '.join(top_sentences)
    except Exception as e:
        st.error(f"Error extracting relevant text: {e}")
        return ""



def extract_shloka_info(text):
    """Extract shloka numbers and their corresponding translations from the text."""
    shloka_pattern = r'(?:^|\s)(\d+(?:\.\d+)?)\s*[-–:]?\s*([^.]*(?:\.[^.]*)*)'
    matches = re.finditer(shloka_pattern, text, re.MULTILINE)
   
    formatted_shlokas = []
    for match in matches:
        shloka_num = match.group(1)
        content = match.group(2).strip()
        if content:  
            formatted_shlokas.append((shloka_num, content))
   
    return formatted_shlokas


def format_relevant_teachings(context, use_gujarati=False):
    """Format the relevant teachings with proper structure and translation."""
    shlokas = extract_shloka_info(context)
   
    formatted_text = []
    for shloka_num, content in shlokas:
        if use_gujarati:
            try:
                translated_content = GoogleTranslator(source='en', target='gu').translate(content)
                formatted_text.append(f"શ્લોક {shloka_num}:\n{translated_content}\n")
            except Exception as e:
                st.error(f"Translation error for shloka {shloka_num}: {e}")
                formatted_text.append(f"શ્લોક {shloka_num}:\n{content}\n")
        else:
            formatted_text.append(f"Shloka {shloka_num}:\n{content}\n")
   
    return "\n".join(formatted_text)


def rag_function(query, vectorstore, llm_model, tokenizer, relevance_model, use_gujarati=False):
    if vectorstore is None:
        return "Error: Vectorstore not initialized.", ""

    try:
        english_query = query if not use_gujarati else translate(query, source='gu', target='en')
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        relevant_docs = retriever.get_relevant_documents(english_query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        relevant_text = llm_extract_relevant_text(english_query, context, tokenizer, relevance_model)
        
        # Translate relevant_text to Gujarati if required
        if use_gujarati:
            relevant_text = translate(relevant_text, source='en', target='gu')
        
        prompt = f"""
        Based on Krishna's teachings, answer the following question using the given context.
        If the answer is not in the context, draw from Krishna's wisdom to provide guidance.
        Respond in a humble and spiritual manner, offering solace and enlightenment to the seeker.
        Include specific references to relevant shlokas when possible.

        Context: {relevant_text}
        Question: {english_query}
        Krishna's answer:
        """
        
        response = llm_model.generate_content(prompt).text
        
        if use_gujarati:
            response = translate(response, source='en', target='gu')
        
        return response, relevant_text  # Now, relevant_text is in Gujarati if use_gujarati=True
        
    except Exception as e:
        st.error(f"Error in RAG function: {e}")
        return "Error retrieving guidance from Krishna.", ""

def main():
    st.title("Krishna Says")
   
    vectorstore = get_vectorstore()
   
    query = st.text_input("Ask Krishna for guidance:")
   
    col1, col2 = st.columns([1, 9])
    with col1:
        use_gujarati = st.checkbox("Ask in Gujarati")
    

    if query:
        st.write("Your question to Krishna:")
        st.markdown(f"**{query}**")
       
        with st.spinner("Krishna is contemplating..."):
            llm_model = genai.GenerativeModel('gemini-pro')
            response, context = rag_function(
                query, vectorstore, llm_model, tokenizer, relevance_model, use_gujarati
            )
       
        st.subheader("Krishna says:")
        st.markdown(
            f'<p style="color: #FFD700; font-style: italic;">{response}</p>',
            unsafe_allow_html=True
        )
       
        with st.expander("View Relevant Shloka", expanded=False):
            st.markdown(
                f"""<div style="background-color: rgba(25, 25, 112, 0.7); border-radius: 10px;
                    padding: 10px; color: white;">{context}</div>""",
                unsafe_allow_html=True
            )
   
    display_footer()

if __name__ == "__main__":
    main()
