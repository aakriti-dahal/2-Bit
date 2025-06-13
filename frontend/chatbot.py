import streamlit as st
import time

# Page setup
st.set_page_config(page_title="E-Governance Legal Chatbot", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .stTextInput>div>div>input {
            border: 1px solid #204939;
            border-radius: 5px;
            padding: 10px 15px;
        }
        .stButton>button {
            background: #204939;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 18px;
        }
        .stButton>button:hover {
            background: #4c8c74;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("E-Governance Legal Chatbot")

# Title & subtitle
st.title("Welcome to the E-Governance Legal Chatbot")
st.markdown("*Your trusted assistant for legal advice and governance matters.*")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    if msg.startswith("**You**:"):
        st.markdown(
            f"<div style='background:#4c8c74;color:white;padding:10px;border-radius:5px;margin:5px 0;max-width:75%;margin-left:auto;'>{msg}</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background:#2b5c49;color:white;padding:10px;border-radius:5px;margin:5px 0;max-width:75%;margin-right:auto;'>{msg}</div>",
            unsafe_allow_html=True)

# Input + send button with icon
col1, col2 = st.columns([0.85, 0.15])
with col1:
    user_input = st.text_input("Ask your question", key="user_input", label_visibility="collapsed", placeholder="Type your query here...")

with col2:
    if st.button("➤", help="Send your message"):
        if user_input.strip():
            st.session_state["messages"].append(f"**You**: {user_input.strip()}")
            time.sleep(1)
            st.session_state["messages"].append("**Chatbot**: Here’s the response to your query.")
            st.session_state["user_input"] = ""  # Clear input
            st.experimental_rerun()

# Sidebar buttons
if st.sidebar.button("Clear Chat History"):
    st.session_state["messages"] = []

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.experimental_rerun()






