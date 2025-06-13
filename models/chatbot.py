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

# Load environment variables
load_dotenv()

# Initialize components
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# pdf_path = "./Features breakdown.pdf"
# print("PDF exists:", os.path.exists(pdf_path))
# print(f"[DEBUG] Loaded {len(pdf_path)} pages from PDF")
# loader = PyPDFLoader(pdf_path)
# docs = loader.load()
# print(f"[DEBUG] Loaded {len(docs)} pages from PDF")
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     add_start_index=True,
# )
# all_splits = text_splitter.split_documents(docs)
#     # ✅ DEBUG PRINT — Make sure your phrase exists
# for i, doc in enumerate(all_splits):
#     print(f"[Chunk {i}]:", doc.page_content[:150])  # Print first 150 chars

    
# Vector Store with persistence
persist_dir = "./chroma_db"
# vector_store = Chroma(
#     embedding_function=embeddings,
#     persist_directory=persist_dir
# )

# Load documents if DB is empty
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    # pdf_path = "https://www.jica.go.jp/Resource/activities/issues/governance/portal/nepal/ku57pq00002khibz-att/civil_code_1st_amendment_en.pdf"
    pdf_path = "./criminal-code-nepal.pdf"
    # print("PDF exists:", os.path.exists(pdf_path))
    # print(f"[DEBUG] Loaded {len(pdf_path)} pages from PDF")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # print(f"[DEBUG] Loaded {len(docs)} pages from PDF")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
     # ✅ DEBUG PRINT — Make sure your phrase exists
    # for i, doc in enumerate(all_splits):
    #     print(f"[Chunk {i}]:", doc.page_content[:150])  # Print first 150 chars
    
    vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_dir
    )

    vector_store.add_documents(documents=all_splits)
    # print(f"[DEBUG] Added {len(all_splits)} documents to vector store")

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
    # print("\n[DEBUG] Retrieved Documents:\n")
    
    # for i, doc in enumerate(retrieved_docs, 1):
    #     print(f"{i}. {doc.page_content[:300]}...\n")
    # return {"context": retrieved_docs}

def generate(state: State):
    # Detect language of the user question
    user_lang = detect(state.question)

    # System prompt based on detected language
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
            "chat_history": state.chat_history
        }

    return {
        "answer": response.content,
        "chat_history": state.chat_history + [
            {"type": "human", "content": state.question},
            {"type": "ai", "content": response.content}
        ]
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
    print("\n🇳🇵 Nepal Legal Assistant (Type 'quit' to exit)\n")
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

if __name__ == "__main__":
    chat_interface()