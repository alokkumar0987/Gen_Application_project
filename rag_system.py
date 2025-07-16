import os
import logging
from typing import Optional, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.retrievers.multi_query import MultiQueryRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseDiagnosisSystem:
    """Specialized RAG system for plant disease diagnosis using PDF resources."""
    
    def __init__(
        self,
        data_file_path: str = "data/simple.pdf",  # Updated to match actual filename
        db_path: str = "chroma_db",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        chunk_size: int = 1500,  # Reduced for better symptom matching
        chunk_overlap: int = 150
    ):
        """
        Initialize plant disease diagnosis system.
        
        Args:
            data_file_path: Path to plant disease PDF
            db_path: Vector database directory
            embedding_model: Embedding model name
            llm_model: LLM model name
            chunk_size: Text chunk size for symptom matching
            chunk_overlap: Overlap between text chunks
        """
        logger.info("Initializing Plant Disease Diagnosis System...")
        self.data_file_path = data_file_path
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        self.model = Ollama(model=self.llm_model_name)
        
        # Enhanced prompt for plant disease diagnosis
        self.prompt_template = """You are a plant pathologist assistant. Use ONLY the context below to:
1. Identify disease symptoms
2. Suggest possible causes
3. Recommend diagnostic steps
4. Provide sample collection guidance

If the answer isn't in the context, say "Based on my training, I don't have information about this specific case. Please consult a plant disease specialist."

CONTEXT:
{context}

QUESTION: {question}

ANSWER STRUCTURE:
- Symptom Identification: 
- Possible Causes: 
- Diagnostic Steps: 
- Sample Collection Advice:"""
        
        self.vector_db = self._prepare_vector_db()

    def _prepare_vector_db(self) -> Chroma:
        """Prepare vector database optimized for symptom matching."""
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            logger.info(f"Loading existing vector DB from {self.db_path}")
            return Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
        
        logger.info(f"Creating new vector DB at {self.db_path}")
        try:
            # Load and process document
            loader = PyPDFLoader(self.data_file_path)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} pages from PDF")
            
            # Optimized splitter for symptom descriptions
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            chunks = splitter.split_documents(docs)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Create vector database
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.db_path
            )
            logger.info(f"Created vector DB with {len(chunks)} chunks")
            return vector_db
            
        except Exception as e:
            logger.error(f"Failed to create vector DB: {e}")
            raise RuntimeError("Vector DB initialization failed") from e

    def _retrieve_context(self, question: str) -> str:
        """Retrieve relevant context with enhanced query handling."""
        try:
            logger.info(f"Processing question: {question}")
            
            # Augment query with botanical terms
            augmented_query = (
                f"{question} Plant pathology terms: symptoms, diagnosis, chlorosis, necrosis, "
                "canker, wilt, rot, pathogen, fungus, bacteria, virus, deficiency"
            )
            
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.vector_db.as_retriever(
                    search_type="mmr",  # Use Max Marginal Relevance
                    search_kwargs={"k": 6}  # Retrieve more documents
                ),
                llm=self.model
            )
            docs = retriever.get_relevant_documents(augmented_query)
            logger.info(f"Retrieved {len(docs)} relevant documents")
            
            # Filter short documents
            filtered_docs = [doc for doc in docs if len(doc.page_content) > 50]
            if not filtered_docs:
                logger.warning("No substantial context retrieved")
                return ""
            
            return "\n\n---\n\n".join(doc.page_content for doc in filtered_docs[:5])  # Limit to top 5
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return ""

    def _format_prompt(self, context: str, question: str) -> str:
        """Format prompt with specialized instructions."""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        return prompt.format(context=context, question=question)

    def answer_query(self, question: str) -> str:
        """
        Answer plant disease questions with diagnostic guidance.
        
        Args:
            question: Plant health query
            
        Returns:
            Structured diagnostic response
        """
        try:
            context = self._retrieve_context(question)
            if not context:
                logger.warning("No relevant context found")
                return ("I couldn't find relevant information in my resources. "
                        "For accurate diagnosis, please consult with your local extension office "
                        "and submit a plant sample.")
                
            prompt = self._format_prompt(context, question)
            logger.debug(f"Generated prompt: {prompt[:500]}...")  # Log partial prompt
            
            response = self.model.invoke(prompt)
            return response
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return ("I'm having trouble processing this request. "
                    "Please try again or contact plant pathology experts directly.")

