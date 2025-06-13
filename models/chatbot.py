import streamlit as st
st.set_page_config(page_title="Nepal Legal Chatbot", layout="wide")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langdetect import detect
from pydantic import BaseModel, Field
from typing import List, Dict
from functools import lru_cache
from gtts import gTTS
import tempfile
import os

# Load env variables
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

@lru_cache(maxsize=100)
def detect_lang_cached(text):
    return detect(text)

class State(BaseModel):
    question: str
    answer: str = ""
    chat_history: List[Dict] = Field(default_factory=list)

def generate_response(state: State):
    lang = detect_lang_cached(state.question)

    if lang == "ne":
        system_prompt = (
            "तपाईं नेपालको संविधानमा आधारित कानुनी सहायक हुनुहुन्छ। "
            "प्रश्नको जवाफ नेपाली भाषामा दिनुहोस्, संविधानको धाराहरू वा अनुच्छेदहरू उद्धरण गर्नुहोस्। "
            "यदि तपाईंलाई जानकारी छैन भने, विनम्रतापूर्वक भन्नुहोस्।"
        )
        tts_lang = "ne"
    else:
        system_prompt = (
            "You are a legal assistant helping users understand Nepal's Constitution. "
            "Please respond in English and cite relevant articles or sections from the Constitution. "
            "If you are unsure, politely say so."
        )
        tts_lang = "en"

    messages = [
        SystemMessage(content=system_prompt),
        *[
            HumanMessage(content=msg["content"]) if msg["type"] == "human" else AIMessage(content=msg["content"])
            for msg in state.chat_history[-4:]
        ],
        HumanMessage(content=state.question)
    ]

    response = llm.invoke(messages)
    text_response = response.content or "I couldn't generate a response. Could you rephrase your question?"

    return text_response, tts_lang

def generate_audio(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmpfile.name)
    return tmpfile.name

# --- UI ---
st.title("🇳🇵 Nepal E-Governance Legal Chatbot")
st.markdown("<h5 style='color:#555;'>Your trusted assistant for constitutional and legal queries</h5>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if msg["type"] == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

if user_input := st.chat_input("Ask your legal question..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"type": "human", "content": user_input})

    with st.spinner("AI is typing..."):
        state = State(question=user_input, chat_history=st.session_state.messages)
        answer_text, tts_lang = generate_response(state)
        audio_file_path = generate_audio(answer_text, tts_lang)

    with st.chat_message("assistant"):
        st.markdown(answer_text)
        st.audio(audio_file_path, format="audio/mp3")
    
    st.session_state.messages.append({
        "type": "ai", 
        "content": answer_text, 
        "audio": audio_file_path
    })
