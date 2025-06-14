import os
import time
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
from googletrans import Translator


# Load environment variables
load_dotenv()

# Initialize components
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# Vector Store with persistence
persist_dir = "./chroma_db"
pdf_paths = [
    "./criminal-code-nepal.pdf",
    "./Constitution-of-Nepal_2072.pdf",
    "./civil_code_1st_amendment_en.pdf",
    "./NP_Criminal Procedure Code_EN.pdf"
]

# Step 1: Only process if DB doesn't exist
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    all_splits = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()

        # Add metadata so you know where each chunk came from
        for doc in docs:
            doc.metadata['source'] = os.path.basename(path)

        # Split each PDF
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(docs)
        all_splits.extend(splits)

    # Create vector store and persist
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    vector_store.add_documents(documents=all_splits)
    print(f"[DEBUG] Indexed {len(all_splits)} chunks from {len(pdf_paths)} files.")

# Step 2: Load vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_dir
)

# Pull RAG prompt
prompt = hub.pull("rlm/rag-prompt")

# Define state with chat history
class State(BaseModel):
    question: str
    context: Optional[List[Document]] = None
    answer: Optional[str] = None
    chat_history: List[Dict] = Field(default_factory=list)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state.question, k=3)#
    print("\n[DEBUG] Retrieved Documents:\n")

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc.page_content[:300]}...\n")
    return {"context": retrieved_docs}

def generate(state: State):
    # Detect language of the user question
    # user_lang = detect(state.question)
    user_lang = st.session_state.get("language", "en")


    # System prompt based on detected language
    if user_lang == "ne":
        system_prompt = (
            "तपाईं नेपालको संविधानमा आधारित कानुनी सहायक हुनुहुन्छ। "
            "प्रश्नको जवाफ नेपाली भाषामा दिनुहोस्, संविधानको धाराहरू वा अनुच्छेदहरू उद्धरण गर्नुहोस्। "
            "यदि तपाईंलाई जानकारी छैन भने, विनम्रतापूर्वक भन्नुहोस्।"
        )
    else:
        system_prompt = (
           "You are a strict legal assistant.explain in details and refers secition and subsection. Only use the following provided excerpts from the constitution to answer. "
            "If the answer is not in the provided context, respond with 'I'm not sure based on the current legal documents.' "
            "Do not use prior knowledge or make assumptions."
        )

    # Format chat history
    conversation_history = "\n".join(
        f"You: {msg['content']}" if msg['type'] == 'human'
        else f"Assistant: {msg['content']}"
        for msg in state.chat_history[-4:]
    )

    legal_context = "\n\n".join(
        f"[Document Excerpt]: {doc.page_content}"
        for doc in state.context
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
            "chat_history": state.chat_history,
            "language": "en"  # Default to English for errors
        }

    return {
        "answer": response.content,
        "chat_history": state.chat_history + [
            {"type": "human", "content": state.question},
            {"type": "ai", "content": response.content}
        ],
        "language": user_lang
    }

# Build the conversation graph
graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
conversation_chain = graph.compile()

def chat_interface():
    """Interactive chat interface"""
    print("\nNepal Legal Assistant (Type 'quit' to exit)\n")
    print("Hello! I can help answer questions about Nepal's legal system. How can I assist you today?\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ('quit', 'exit'):
                print("\nThank you for using Satyanista. Have a great day!")
                break

            if not user_input:
                print("Please enter a question.")
                continue

            # Process the question
            result = conversation_chain.invoke({
                "question": user_input,
                "chat_history": history
            })

            # Update history
            history = result["chat_history"]

            # Print formatted response
            print(f"\nAssistant: {result['answer']}\n")

        except KeyboardInterrupt:
            print("\n\nSession ended by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nSorry, I encountered an error: {str(e)}")
            continue


def generate_audio(text, lang_code):
    if lang_code == "ne":
        # Translate English answer to Nepali for speaking
        translator = Translator()
        try:
            translated_text = translator.translate(text, dest='ne').text
        except Exception as e:
            print(f"Translation error: {e}")
            translated_text = text  # fallback to English
        tts = gTTS(text=translated_text, lang='ne')
    else:
        tts = gTTS(text=text, lang='en')

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmpfile.name)
    return tmpfile.name


# --- UI ---
st.title("🇳🇵 Nepal E-Governance Legal Chatbot")
st.markdown("<h5 style='color:#555;'>Your trusted assistant for constitutional and legal queries</h5>", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []

with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("### 🌐 Language")
    language_choice = st.radio("Select language for both reply and audio:", ["English", "Nepali"], index=0)

    # Save to session state
    st.session_state.language = "ne" if language_choice == "Nepali" else "en"

# Display chat history
for msg in st.session_state.messages:
    role = "user" if msg["type"] == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# Handle user input
if user_input := st.chat_input("Ask your legal question..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process response
    try:
        # Step 1: Show initial processing
        with st.status("Processing your question...", expanded=True) as status:
            # Create initial state
            current_state = State(
                question=user_input,
                chat_history=st.session_state.chat_history
            )
            
            # Step 2: Show document retrieval
            status.update(label="🔍 Retrieving relevant documents...")
            result = conversation_chain.invoke(current_state.dict())
            retrieved_docs = result.get("context", [])
            
            # Display chunks one by one
            status.update(label="📑 Analyzing document chunks...")
            for i, doc in enumerate(retrieved_docs):
                st.write(f"Analyzing chunk {i+1}...")
                st.markdown(f"```\n{doc.page_content[:150]}...\n```")
                time.sleep(1)  # Add slight delay for visibility
            
            # Step 3: Generate response
            status.update(label="🤖 Generating response...")
            answer_text = result["answer"]
            tts_lang = result.get("language", "en")
            
            # Step 4: Generate audio
            status.update(label="🎵 Creating audio response...")
            
            audio_file_path = generate_audio(
            answer_text, 
            st.session_state.language  #  this exists
             )
                    

            
            status.update(label="✅ Done!", state="complete")

        # Update chat history and display response
        st.session_state.chat_history = result["chat_history"]
        st.session_state.messages.append({"type": "human", "content": user_input})
        st.session_state.messages.append({
            "type": "ai",
            "content": answer_text,
            "audio": audio_file_path
        })
        
        # Display final response
        import time

        with st.chat_message("assistant"):
            st.markdown(answer_text)

            # Show loading spinner before displaying audio
            with st.spinner("🎵 Preparing audio response..."):
                # Optional delay so user sees the spinner
                time.sleep(1.5)  # Adjust as needed
                if os.path.exists(audio_file_path):
                    st.audio(audio_file_path, format="audio/mp3")
                else:
                    st.warning("⚠️ Audio is not ready yet. Please try again.")

            
    except Exception as e:
        st.error(f"Error: {str(e)}")