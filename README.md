# 🧠 Gen_Application_project

A powerful *Retrieval-Augmented Generation (RAG)* system with *Ollama LLM* and *Streamlit UI* for querying PDF documents.  
Upload PDFs, ask questions, and get instant answers in a clean web app.

---

## 📸 Demo Screenshots
<img width="1920" height="1080" alt="Screenshot 2025-07-16 153135" src="https://github.com/user-attachments/assets/aa288fd4-3374-49c3-b88f-6aaf293c561a" />


<img width="1920" height="1080" alt="Screenshot 2025-07-16 153331" src="https://github.com/user-attachments/assets/8dd8f6bd-fd50-45d8-b4b0-67485813727f" />

---

## 🚀 Features
✅ Load and process PDF documents  
✅ Split text into smart chunks for better context  
✅ Store embeddings in a Chroma Vector Database  
✅ Query using LangChain + Ollama LLM (e.g., Llama 3.2)  
✅ Streamlit Web App for easy Q&A  
✅ Admit card PDFs in the data/ folder and Chatbot using local llm model   llama3.2

---
## 📥 Installation

### Clone Repository
```bash
git clone https://github.com/alokkumar0987/Gen_Application_project.git
cd Gen_Application_project


# create virtual enviroment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


pip install -r requirements.txt

## RUN THE App
streamlit run app.py

