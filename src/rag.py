from typing import Dict, List
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.config import AppConfig
from src.vectorstore import VectorManager
from src.logger import logger
import requests

class RAGEngine:
    """
    Orchestrates Retrieval-Augmented Generation.
    
    Per AGENTCONTEXT.md:
    - Explicit failure modes (Ollama check)
    - Clear data flow (Retrieve -> Prompt -> Generate)
    - Must cite sources
    """
    
    def __init__(self, config: AppConfig, vector_manager: VectorManager):
        self.config = config
        self.vector_manager = vector_manager
        
        # Verify Ollama is reachable
        self._check_ollama()
        
        # Initialize Ollama client
        logger.info(f"Connecting to Ollama: {config.llm.model}")
        self.llm = Ollama(
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            num_predict=config.llm.context_window  # Respect hardware limit
        )
        
        # Build prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=self._build_prompt()
        )
    
    def _check_ollama(self):
        """Fail fast if Ollama is not running."""
        try:
            response = requests.get(f"{self.config.llm.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError("Ollama API returned non-200 status")
            logger.debug("Ollama service is reachable")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot reach Ollama at {self.config.llm.base_url}")
            logger.error("Fix: Ensure Ollama is running (run `ollama serve`)")
            raise RuntimeError("Ollama service not available") from e
    
    def _build_prompt(self) -> str:
        """Construct system prompt per AGENTCONTEXT.md standards."""
        return """You are a precision information retrieval assistant.

Context from documents:
{context}

User Question:
{question}

Instructions:
- Answer ONLY using the provided context.
- If the context does not contain the answer, state: "The provided documents do not contain this information."
- Cite sources by filename when possible.
- Be concise and factual.

Answer:"""
    
    def query(self, question: str) -> Dict[str, any]:
        """
        Execute RAG query.
        
        Returns:
            Dict with 'answer' and 'sources'
        """
        # Step 1: Retrieve relevant documents
        k = self.config.retrieval.k_final
        logger.debug(f"Retrieving top {k} documents for: {question}")
        retrieved_docs = self.vector_manager.search(question)[:k]
        
        if not retrieved_docs:
            logger.warning("No documents retrieved from vector store")
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "sources": []
            }
        
        # Step 2: Build context
        context = self._format_context(retrieved_docs)
        
        # Step 3: Generate answer
        prompt = self.prompt_template.format(context=context, question=question)
        logger.debug(f"Prompt length: {len(prompt)} chars")
        
        try:
            answer = self.llm.invoke(prompt)
            return {
                "answer": answer,
                "sources": self._extract_sources(retrieved_docs)
            }
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError("Failed to generate answer") from e
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved docs into LLM context."""
        chunks = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown")
            chunks.append(f"[Source {i}: {source}]\n{doc.page_content}\n")
        return "\n".join(chunks)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict]:
        """Extract citation metadata."""
        sources = []
        for doc in documents:
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", -1)
            })
        return sources
