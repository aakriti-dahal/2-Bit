import streamlit as st

st.set_page_config(page_title="Chatbot", page_icon="💬")

st.title("💬 Chatbot")
st.caption("🧠 Chat interface powered by your custom backend (LangChain + ChromaDB)")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

# Append user input and placeholder bot response
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder response (static for now, to be replaced with actual logic)
    bot_response = f"🤖 (Response placeholder for: '{user_input}')"
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)