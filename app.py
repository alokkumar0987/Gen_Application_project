import streamlit as st
from rag_system import RAGSystem

rag_system = RAGSystem()

def main():
    st.set_page_config(page_title="RAG AMA", page_icon="🤖")
    st.title("📄 Ask Me Anything - RAG System")
    st.write("Ask questions based on the PDF content.")

    question = st.text_input("💬 Enter your question:")
    if st.button("Ask"):
        if question.strip() == "":
            st.warning("⚠ Please enter a question.")
        else:
            with st.spinner("🔎 Searching for answer..."):
                response = rag_system.answer_query(question)
            st.success("✅ Answer:")
            st.write(response)

if __name__== "__main__":
    main()
