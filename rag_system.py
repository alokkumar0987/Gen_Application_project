from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate

class RAGSystem:
    def __init__(self, data_dir="data", db_path="chroma", model_name="llama3.2"):
        print("🚀 Initializing RAG System...")
        self.data_dir = data_dir
        self.db_path = db_path
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vectordb = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.llm = Ollama(model=model_name, temperature=0.7)
        self.prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, say "I don't know". Don't make up answers.

        {context}

        Question: {question}
        Answer:
        """
        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        pages = self.load_pdfs()
        chunks = self.split_documents(pages)
        chunks = self.assign_chunk_ids(chunks)
        self.add_chunks_to_db(chunks)

    def load_pdfs(self):
        print(f" Loading PDFs from directory: {self.data_dir}")
        loader = PyPDFDirectoryLoader(self.data_dir)
        pages = loader.load()
        print(f" Loaded {len(pages)} documents")
        return pages

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        print("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def assign_chunk_ids(self, chunks):
        print(" Assigning unique IDs to chunks...")
        prev_page_id, curr_chunk_index = None, 0
        for chunk in chunks:
            src, page = chunk.metadata.get("source"), chunk.metadata.get("page")
            curr_page_id = f"{src}_{page}"
            curr_chunk_index = curr_chunk_index + 1 if curr_page_id == prev_page_id else 0
            chunk.metadata["chunk_id"] = f"{curr_page_id}_{curr_chunk_index}"
            prev_page_id = curr_page_id
        return chunks

    def add_chunks_to_db(self, chunks):
        print("Adding chunks to Vector DB...")
        present_in_db = self.vectordb.get()
        ids_in_db = present_in_db.get("ids", [])
        new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in ids_in_db]
        print(f"New chunks to add: {len(new_chunks)}")
        if new_chunks:
            self.vectordb.add_documents(
                new_chunks,
                ids=[c.metadata["chunk_id"] for c in new_chunks]
            )
            print(" Chunks added to DB.")
        else:
            print("All chunks already in DB.")

    def answer_query(self, question):
        print(f"\n❓ Query: {question}")
        context = self.vectordb.similarity_search_with_score(question, k=5)
        if not context:
            return "⚠ No relevant context found."
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in context])
        prompt = ChatPromptTemplate.from_template(self.prompt_template).format(
            context=context_text, question=question)
        response = self.llm.invoke(prompt)
        return response
