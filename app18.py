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


load_dotenv()
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAQmgOq7z-n3yCotriI6-W3wpzIDap6Xqg'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

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
    }
    .stMarkdown {
        color: #FFD700;
    }
    .css-1wbqy5l {
        background-color: rgba(25, 25, 112, 0.7);
    }
    .stButton {
        background-color: rgba(255, 215, 0, 0.8);
        border: none;
        border-radius: 10px;
    }
    .stExpanderHeader {
        font-weight: bold;
        color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

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
    # pdf_path = "/content/translationofbg.pdf" 
    pdf_path = "https://raw.githubusercontent.com/yess-wee/KrishnaSaysModel/main/translationofbg.pdf"

    if os.path.exists(faiss_index_path):
        try:
            st.info("Loading existing FAISS index...")
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


        sentences = context.split('.')
        sentence_scores = []
        for sentence in sentences:
            inputs = tokenizer(query, sentence, return_tensors="pt", truncation=True, max_length=512, padding=True)
            outputs = model(**inputs)
            score = torch.nn.functional.softmax(outputs.logits, dim=1)[0, 1].item()
            sentence_scores.append((sentence, score))


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
       
        formatted_context = format_relevant_teachings(context, use_gujarati)
       
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
       
        return response, formatted_context
       
    except Exception as e:
        st.error(f"Error in RAG function: {e}")
        return "Error processing your question.", ""


def main():
    st.title("Krishna Says")
   
    vectorstore = get_vectorstore()
  
    st.sidebar.title("Language Selection")
    use_gujarati = st.sidebar.checkbox("Ask in Gujarati")

    query = st.text_input(
        "Ask Krishna for guidance:" if not use_gujarati else "તમારો પ્રશ્ન ગુજરાતીમાં પૂછો:",
        ""
    )
   
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
                f"""<div style="background-color: rgba(0, 0, 0, 0.5); padding: 20px; border-radius: 10px;">
                    <pre style="color: #FFD700; white-space: pre-wrap;">{context}</pre>
                </div>""",
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()


