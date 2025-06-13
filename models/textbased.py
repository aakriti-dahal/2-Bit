import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import StateGraph, START
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langdetect import detect
from gtts import gTTS
import tempfile
import os
import streamlit as st
import time

# Load environment variables
load_dotenv()

# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# Vector Store with persistence
persist_dir = "./chroma_db"

if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    pdf_path = "./criminal-code-nepal.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[DEBUG] Loaded {len(docs)} pages from PDF")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    vector_store.add_documents(documents=all_splits)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_dir
)

# Pull RAG prompt (if needed)
prompt = hub.pull("rlm/rag-prompt")

# Define state with chat history
class State(BaseModel):
    question: str
    context: Optional[List] = None
    answer: Optional[str] = None
    chat_history: List[Dict] = Field(default_factory=list)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state.question, k=3)
    print("\n[DEBUG] Retrieved Documents:\n")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc.page_content[:300]}...\n")
    return {"context": retrieved_docs}

def generate(state: State):
    # Detect user language
    user_lang = detect(state.question)

    if user_lang == "ne":
        system_prompt = (
            "तपाईं नेपालको संविधानमा आधारित कानुनी सहायक हुनुहुन्छ। "
            "प्रश्नको जवाफ नेपाली भाषामा दिनुहोस्, संविधानको धाराहरू वा अनुच्छेदहरू उद्धरण गर्नुहोस्। "
            "यदि तपाईंलाई जानकारी छैन भने, विनम्रतापूर्वक भन्नुहोस्।"
        )
    else:
        system_prompt = (
            "You are a strict legal assistant. Only use the following provided excerpts from the constitution to answer. "
            "If the answer is not in the provided context, respond with 'I'm not sure based on the current legal documents.' "
            "Do not use prior knowledge or make assumptions."
        )

    conversation_history = "\n".join(
        f"Human: {msg['content']}" if msg['type'] == 'human' else f"AI: {msg['content']}"
        for msg in state.chat_history[:-1]
    )
    legal_context = "\n\n".join(
        f"[Document Excerpt]: {doc.page_content}"
        for doc in (state.context or [])
    )

    messages = [
        SystemMessage(content=system_prompt),
        *[
            HumanMessage(content=msg['content']) if msg['type'] == 'human'
            else AIMessage(content=msg['content'])
            for msg in state.chat_history[-10:]
        ],
        HumanMessage(content=f"""Based on our conversation so far:
{conversation_history}

And these relevant legal excerpts:
{legal_context}

Please answer this new question: {state.question}

Respond conversationally but accurately, citing sources when possible. Give the article number or section from the constitution if applicable and give the possible sentence for the said crime/issue if available in the document. If you don't know, say so politely.""")
    ]

    response = llm.invoke(messages)

    if not response.content:
        return {
            "answer": "I couldn't generate a response. Could you rephrase your question?",
            "chat_history": state.chat_history
        }

    return {
        "answer": response.content,
        "chat_history": state.chat_history + [
            {"type": "human", "content": state.question},
            {"type": "ai", "content": response.content}
        ]
    }

# Build conversation graph
graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
conversation_chain = graph.compile()

def generate_ui_response(state: State):
    if not state:
        return "", "en"
    
    result = conversation_chain.invoke({
        "question": state.question,
        "chat_history": state.chat_history
    })

    response_text = result.get('answer', 'Sorry, no answer generated.')
    language_code = "en"  # You can adjust based on detect or other logic
    
    return response_text, language_code

# --- Streamlit UI ---

st.title("🇳🇵 Nepal E-Governance Legal Chatbot")
st.markdown("<h5 style='color:#555;'>Your trusted assistant for constitutional and legal queries</h5>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if user_input := st.chat_input("Ask your legal question..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"type": "human", "content": user_input})

    with st.spinner("AI is typing..."):
        state = State(question=user_input, chat_history=st.session_state.messages)
        # Retrieve relevant documents
        retrieved = retrieve(state)
        retrieved_docs = retrieved.get("context", [])
        # Show retrieved documents as "AI is thinking..." before generating answer
        with st.chat_message("assistant"):
            # Add a loading spinner for delay effect
            # with st.spinner("Retrieving relevant legal excerpts..."):
                st.markdown("🤔 Retrieving relevant legal excerpts...", unsafe_allow_html=True)
                time.sleep(5.5)  # Simulate delay for better UX
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(
                        f"<div style='font-family: monospace; padding: 0.5em; border-radius: 6px; margin-bottom: 0.5em;'>"
                        f"<b>Excerpt {i}:</b> {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # Generate answer and show to user
        # Update state with retrieved context before generating
        state.context = retrieved_docs
        answer_text, tts_lang = generate_ui_response(state)

    with st.chat_message("assistant"):
        st.markdown(answer_text)

    st.session_state.messages.append({
        "type": "ai",
        "content": answer_text
    })

        
    