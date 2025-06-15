import streamlit as st
from utils.utils import sidebar_logo


def feature_card(title, description, emoji):
    st.markdown(f"""
    <div style="
        background-color: #f0f2f6;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 2px 2px 12px #dcdde1;
        ">
        <h3 style="color:#2c3e50;">{emoji} {title}</h3>
        <p style="color:#34495e;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Satyanistha", page_icon="⚖️", layout="centered")

    st.title("⚖️ satyanistha")
    st.write("Your AI-powered assistant for navigating Nepal's legal system.")

    # st.image("https://images.unsplash.com/photo-1573164574397-0d46a20e18e7?auto=format&fit=crop&w=800&q=80",
    #          caption="Legal assistance powered by AI",
    #           use_container_width=True)

    st.markdown("---")

    st.header("Future goals")

    cols = st.columns(2)

    with cols[0]:
        feature_card(
            "Legal Chat",
            "Get answers to your legal questions based on Nepal law, anytime.",
            "💬"
        )
        feature_card(
            "Recommendation System",
            "Receive personalized legal advice and document suggestions.",
            "🔍"
        )
        feature_card(
            "Summarize Laws",
            "Get easy-to-understand summaries of complex legal documents.",
            "📄"
        )

    with cols[1]:
        feature_card(
            "Book Appointments",
            "Schedule a meeting with a legal expert.",
            "📅"
        )
        feature_card(
            "Solve Legal Queries",
            "Ask about contracts, disputes, rights, and more.",
            "⚖️"
        )

    sidebar_logo()
    st.markdown("---")

    st.header("Get Started")

    option = st.selectbox("Select a service", [
        "Chat with Legal Assistant",
        "Book an Appointment",
        "Get Legal Recommendations",
        "Ask Legal Queries",
        "Summarize a Law Document"
    ])

    if option == "Chat with Legal Assistant":
        st.info("👉 This will launch the legal chat interface (coming soon).")
    elif option == "Book an Appointment":
        st.info("👉 Appointment booking system coming soon.")
    elif option == "Get Legal Recommendations":
        st.info("👉 Personalized recommendations will be available here.")
    elif option == "Ask Legal Queries":
        st.info("👉 Ask your legal questions and get answers.")
    elif option == "Summarize a Law Document":
        st.info("👉 Upload a law document to get a summary.")

    st.markdown("---")

    st.write("© 2025 Nepal Legal Chatbot. All rights reserved.")

if __name__ == "__main__":
    main()
