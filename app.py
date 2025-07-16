import streamlit as st
from rag_system import PlantDiseaseDiagnosisSystem  # Make sure the class is in rag_system.py

@st.cache_resource
def load_system():
    # Initialize the system with the correct PDF
    return PlantDiseaseDiagnosisSystem(data_file_path="data/e1592.pdf")

rag_system = load_system()

def main():
    st.title("🌱 Plant Disease Diagnosis Assistant")
    st.markdown("Describe plant symptoms or ask questions about plant diseases.")
    
    question = st.text_area("Describe the symptoms or ask a question:", height=150)
    
    if st.button("Get Diagnosis"):
        with st.spinner("Analyzing symptoms and consulting resources..."):
            response = rag_system.answer_query(question)
        
        st.subheader("Diagnosis Report")
        st.markdown(response)  # Display as markdown

if __name__ == "__main__":
    main()