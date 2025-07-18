# 🧠 Gen_Application_project

A powerful *Retrieval-Augmented Generation (RAG)* system with *Ollama LLM* and *Streamlit UI* for querying PDF documents.  
Upload PDFs, ask questions, and get instant answers in a clean web app.

---



## 🏗️ Architecture of RAG System

📂 **Step 1:** Load PDF  
⬇  
🪓 **Step 2:** Split PDF into smaller text chunks  
⬇  
🆔 **Step 3:** Assign unique IDs to each chunk  
⬇  
🔗 **Step 4:** Generate embeddings & store them in Chroma Vector DB  
⬇  
🤖 **Step 5:** User asks questions via Streamlit Chatbot UI  
⬇  
💬 **Step 6:** Local Llama 3.2 model (Ollama) retrieves context & generates answer



## 📸 Demo Screenshots
<img width="1920" height="1080" alt="Screenshot 2025-07-16 153135" src="https://github.com/user-attachments/assets/aa288fd4-3374-49c3-b88f-6aaf293c561a" />


<img width="1920" height="1080" alt="Screenshot 2025-07-16 153331" src="https://github.com/user-attachments/assets/8dd8f6bd-fd50-45d8-b4b0-67485813727f" />

<img width="1920" height="1080" alt="Screenshot 2025-07-16 153608" src="https://github.com/user-attachments/assets/7a20a11a-8b53-4c0c-8e23-daf12adf4a94" />


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

#Run the App
streamlit run app.py

```
## 🙋‍♂ Author
👤 *Alok Kumar*  
🌐 [GitHub](https://github.com/alokkumar0987) | [LinkedIn](https://www.linkedin.com/in/alok-kumar-784025309)

---

## 📚 Tech Stack
🚀 *Languages & Frameworks*
- 🐍 Python 3.9+
- 🖥 Streamlit (Frontend Web UI)
- 🧠 LangChain (RAG Pipeline)
- 📝 Ollama (LLMs: LLaMA 3.2)
- 🗃 ChromaDB (Vector Database)

📦 *Tools*
- 💻 VS Code
- 🐙 Git & GitHub
- 📦 Virtual Environments (venv)
- 📑 PDF Processing with LangChain



[![LinkedIn](https://img.shields.io/badge/LinkedIn-AlokKumar-blue?logo=linkedin)](https://www.linkedin.com/in/alok-kumar-784025309)
