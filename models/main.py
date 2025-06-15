import os
import hashlib
import threading
import time
from typing import List, Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from utils.utils import sidebar_logo

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain import hub
from langgraph.graph import StateGraph, START
from gtts import gTTS
from googletrans import Translator

# --- Load environment variables ---
load_dotenv()

# --- Constants ---
PERSIST_DIR = "./chroma_db"
INDEX_FLAG_PATH = os.path.join(PERSIST_DIR, ".index_complete")
AUDIO_CACHE_DIR = "audio_cache"

# --- Data Model for chat state ---
class State(BaseModel):
    question: str
    context: Optional[List[Document]] = None
    answer: Optional[str] = None
    chat_history: List[Dict] = Field(default_factory=list)

# --- Cache expensive resources once ---
@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

@st.cache_resource(show_spinner=False)
def load_vector_store():
    return Chroma(embedding_function=load_embeddings(), persist_directory=PERSIST_DIR)

llm = load_llm()
vector_store = load_vector_store()

# --- Indexing PDF files (run only if index not complete) ---
def build_index_if_needed():
    if not os.path.exists(INDEX_FLAG_PATH):
        pdf_paths = [
            "./criminal-code-nepal.pdf",
            "./Constitution-of-Nepal_2072.pdf",
            "./civil_code_1st_amendment_en.pdf",
            "./NP_Criminal Procedure Code_EN.pdf"
        ]
        all_splits = []

        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = os.path.basename(path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
            all_splits.extend(splitter.split_documents(docs))

        temp_vs = Chroma(embedding_function=load_embeddings(), persist_directory=PERSIST_DIR)
        temp_vs.add_documents(documents=all_splits)

        # Mark index complete
        os.makedirs(PERSIST_DIR, exist_ok=True)
        with open(INDEX_FLAG_PATH, "w") as f:
            f.write("done")

build_index_if_needed()

# --- Prompt from hub ---
prompt = hub.pull("rlm/rag-prompt")

# --- Functions for retrieval and generation ---
def retrieve(state: State):
    docs = vector_store.similarity_search(state.question, k=3)
    return {"context": docs}

def generate(state: State):
    user_lang = st.session_state.get("language", "en")

    if user_lang == "ne":
        system_prompt = (
            "तपाईं नेपालको संविधानमा आधारित कानुनी सहायक हुनुहुन्छ। "
            "प्रश्नको जवाफ नेपाली भाषामा दिनुहोस्, संविधानको धाराहरू वा अनुच्छेदहरू उद्धरण गर्नुहोस्। "
            "यदि तपाईंलाई जानकारी छैन भने, विनम्रतापूर्वक भन्नुहोस्।"
        )
    else:
        system_prompt = (
            "You are a legal assistant trained on Nepal's constitution and criminal code."
            " Use ONLY the context excerpts provided to answer the user's legal question clearly and thoroughly."
            " Your response should:"
            "\n- Reference relevant sections and subsections"
            "\n- Explain in simple legal language"
            "\n- If punishment or legal consequences are asked, explain them if found in context"
            "\n- DO NOT answer if context does not support it — instead, respond: 'The provided legal documents do not contain enough information to answer this question confidently.'"

       )

    # Prepare messages to LLM
    messages = [
        SystemMessage(content=system_prompt),
        *[
            HumanMessage(content=msg['content']) if msg['type'] == 'human'
            else AIMessage(content=msg['content'])
            for msg in state.chat_history[-10:]
        ],
        HumanMessage(content=f"""Based on our conversation so far:
{chr(10).join(f'You: {msg["content"]}' if msg['type']=='human' else f'Assistant: {msg["content"]}' for msg in state.chat_history[-4:])}

And these relevant legal excerpts:
{chr(10).join(f'[Excerpt]: {doc.page_content[:500]}' for doc in state.context)}

Please answer this question: {state.question}
""")
    ]

    response = llm.invoke(messages)
    return {
        "answer": response.content or "I couldn't generate a response. Could you rephrase?",
        "chat_history": state.chat_history + [
            {"type": "human", "content": state.question},
            {"type": "ai", "content": response.content}
        ],
        "language": user_lang
    }

# --- StateGraph setup ---
graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
conversation_chain = graph.compile()

def generate_audio_blocking(text, lang_code):
    path = get_audio_cache_path(text, lang_code)
    if not os.path.exists(path):
        with st.spinner("🎵 Generating audio, please wait..."):
            # This call will block until the file is saved
            if lang_code == "ne":
                try:
                    translated = Translator().translate(text, dest='ne').text
                except Exception:
                    translated = text
                gTTS(text=translated, lang='ne').save(path)
            else:
                gTTS(text=text, lang='en').save(path)
            time.sleep(0.5)  # extra safety delay
    return path
# --- Audio generation helpers ---
def get_audio_cache_path(text: str, lang_code: str) -> str:
    text_hash = hashlib.md5((text + lang_code).encode()).hexdigest()
    return os.path.join(AUDIO_CACHE_DIR, f"{text_hash}.mp3")

def get_audio_cache_path(text, lang_code):
    text_hash = hashlib.md5((text + lang_code).encode()).hexdigest()
    return os.path.join("audio_cache", f"{text_hash}.mp3")

def generate_audio(text, lang_code):
    path = get_audio_cache_path(text, lang_code)

    def _generate():
        try:
            os.makedirs("audio_cache")
            if lang_code == "ne":
                try:
                    translated = Translator().translate(text, dest='ne').text
                except Exception:
                    translated = text
                gTTS(text=translated, lang='ne').save(path)
            else:
                gTTS(text=text, lang='en').save(path)
        except Exception as e:
            print(f"Audio generation error: {e}")

    if not os.path.exists(path):
        threading.Thread(target=_generate).start()

    return path
# Run audio generation in background thread (non-blocking)
def generate_audio_async(text: str, lang_code: str):
    def worker():
        path = generate_audio(text, lang_code)
        if not path:
            print(f"[WARN] Audio generation failed for text: {text[:50]}...")

    threading.Thread(target=worker, daemon=True).start()

# --- Streamlit UI ---

st.set_page_config(page_title="Nepal Legal Chatbot", page_icon="⚖️", layout="wide")


sidebar_logo()

st.title("\U0001F1F3\U0001F1F5 सत्यनिष्ठ")
st.markdown("<h5 style='color:#555;'>Your trusted assistant for legal queries</h5>", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    st.session_state.language = "en"

# Sidebar for controls
with st.sidebar:
    st.header("\u2699\ufe0f Options")
    if st.button("\U0001F9F9 Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.experimental_rerun()

    st.markdown("### \U0001F310 Language")
    language_choice = st.radio("Select language:", ["English", "Nepali"], index=0 if st.session_state.language == "en" else 1)
    st.session_state.language = "ne" if language_choice == "Nepali" else "en"

# Display chat messages from history
for msg in st.session_state.messages:
    role = "user" if msg["type"] == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# User input prompt
user_input = st.chat_input("Ask your legal question...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # Build current state and invoke conversation chain
        current_state = State(
            question=user_input,
            chat_history=st.session_state.chat_history
        )
        with st.spinner("Processing..."):
            result = conversation_chain.invoke(current_state.dict())

        # Extract data
        answer_text = result["answer"]
        tts_lang = result.get("language", "en")

        # Update chat states
        st.session_state.chat_history = result["chat_history"]
        st.session_state.messages.append({"type": "human", "content": user_input})
                
                # Display assistant message immediately (without audio)
        with st.chat_message("assistant"):
            st.markdown(answer_text)

        # Generate audio with spinner, blocking call
        audio_file_path = generate_audio_blocking(answer_text, tts_lang)

        # Save AI message with audio
        st.session_state.messages.append({
            "type": "ai",
            "content": answer_text,
            "audio": audio_file_path
        })

        # Play the audio now that it's ready
        st.audio(audio_file_path, format="audio/mp3")

    except Exception as e:
        st.error(f"Error: {e}")
