from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import AppConfig
from src.logger import logger
from src.extractor import MetadataExtractor

class DocumentLoader:
    """
    Handles loading and preprocessing of documents from various formats.
    
    Features:
    - Deterministic chunking
    - Explicit failure modes
    - Strict config usage
    - Intelligent Metadata Extraction (Ollama)
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.chunk_size = config.retrieval.chunk_size
        self.chunk_overlap = config.retrieval.chunk_overlap
        self.valid_extensions = set(config.ingestion.valid_extensions)
        
        # Initialize Extractor
        self.extractor = MetadataExtractor(config)
        
        # Initialize splitter deterministically
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "] # Order matters for semantic preservation
        )
        
        logger.debug(f"DocumentLoader initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def load_file(self, file_path: Path) -> List[Any]:
        """
        Load a single file with rigorous validation.
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = file_path.suffix.lower()
        if ext not in self.valid_extensions:
            logger.warning(f"Skipping unsupported file: {file_path.name}")
            return []
            
        try:
            loader = self._get_loader(file_path, ext)
            documents = loader.load()
            
            if not documents:
                return []

            # Extract Metadata using LLM (only needs first chunk of text)
            full_text = "\n".join([doc.page_content for doc in documents])
            extracted_metadata = self.extractor.extract(full_text, file_path.name)
            
            # Application: Normalize metadata onto all docs
            for doc in documents:
                doc.metadata["source"] = str(file_path.absolute())
                doc.metadata["filename"] = file_path.name
                # Merge extracted metadata
                doc.metadata.update(extracted_metadata)
                
            return documents
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise RuntimeError(f"Document loading failed for {file_path}") from e

    def _get_loader(self, path: Path, ext: str):
        """Factory method for loaders."""
        if ext == ".pdf":
            return PyPDFLoader(str(path))
        elif ext == ".docx":
            return Docx2txtLoader(str(path))
        elif ext in [".txt", ".md"]:
            return TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"No loader for {ext}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into chunks with stable IDs.
        """
        if not documents:
            return []
            
        chunks = self.text_splitter.split_documents(documents)
        
        # Assign deterministic metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
        logger.info(f"Split {len(documents)} docs into {len(chunks)} chunks")
        return chunks
