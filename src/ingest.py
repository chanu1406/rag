"""
Document Ingestion Module
Handles loading, parsing, and chunking of documents for RAG.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """
    Handles loading and preprocessing of documents from various formats.

    Supports: PDF, TXT, MD, DOCX
    Optimized for efficient processing before embedding on RTX 4060.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DocumentLoader with configuration.

        Args:
            config: Configuration dictionary containing document processing parameters
        """
        self.config = config
        self.chunk_size = config.get('document_processing', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('document_processing', {}).get('chunk_overlap', 200)
        self.supported_formats = config.get('document_processing', {}).get('supported_formats', ['pdf', 'txt', 'md', 'docx'])

        # Initialize text splitter with config parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=config.get('document_processing', {}).get('separators', ["\n\n", "\n", ". ", " "])
        )

        logger.info(f"DocumentLoader initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def load_single_document(self, file_path: str) -> List[Any]:
        """
        Load a single document from file path.

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects (LangChain format)

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path_obj = Path(file_path)

        # Validate file exists
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Determine file extension
        extension = file_path_obj.suffix.lower().lstrip('.')

        # Check if format is supported
        if extension not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: .{extension}\n"
                f"Supported formats: {', '.join(self.supported_formats)}"
            )

        logger.info(f"Loading document: {file_path} (format: {extension})")

        # Route to appropriate loader based on extension
        try:
            if extension == 'pdf':
                loader = PyPDFLoader(file_path)
            elif extension == 'txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif extension == 'md':
                # For markdown, use TextLoader (simpler and more reliable)
                loader = TextLoader(file_path, encoding='utf-8')
            elif extension == 'docx':
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported format: {extension}")

            # Load documents
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} page(s) from {file_path_obj.name}")
            return documents

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to load document: {str(e)}")

    def load_directory(self, directory_path: str, glob_pattern: str = "**/*") -> List[Any]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            glob_pattern: Glob pattern for file matching (default: all files)

        Returns:
            List of Document objects from all files
        """
        dir_path = Path(directory_path)

        # Validate directory exists
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        logger.info(f"Loading documents from directory: {directory_path}")

        # Find all files matching supported formats
        all_documents = []
        files_loaded = 0
        files_failed = 0

        # Iterate through supported formats
        for ext in self.supported_formats:
            pattern = f"**/*.{ext}" if glob_pattern == "**/*" else glob_pattern
            matching_files = list(dir_path.glob(pattern))

            for file_path in matching_files:
                try:
                    docs = self.load_single_document(str(file_path))
                    all_documents.extend(docs)
                    files_loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {str(e)}")
                    files_failed += 1

        logger.info(
            f"Directory loading complete: {files_loaded} files loaded, "
            f"{files_failed} files failed, {len(all_documents)} total pages"
        )

        return all_documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks for embedding.

        Args:
            documents: List of Document objects to split

        Returns:
            List of chunked Document objects
        """
        if not documents:
            logger.warning("No documents to split")
            return []

        logger.info(f"Splitting {len(documents)} documents into chunks...")

        # Apply text splitter
        split_docs = self.text_splitter.split_documents(documents)

        # Add chunk metadata
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_index'] = i
            doc.metadata['total_chunks'] = len(split_docs)

        logger.info(
            f"Splitting complete: {len(documents)} documents -> {len(split_docs)} chunks "
            f"(avg {len(split_docs)//len(documents) if documents else 0} chunks/doc)"
        )

        return split_docs

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text before chunking.

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned text
        """
        import re
        import unicodedata

        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Collapse multiple newlines

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def load_and_split(self, source: str, is_directory: bool = True) -> List[Any]:
        """
        Convenience method to load and split documents in one call.

        Args:
            source: File path or directory path
            is_directory: Whether source is a directory

        Returns:
            List of split Document objects ready for embedding
        """
        logger.info(f"Starting load and split pipeline for: {source}")

        # Load documents
        if is_directory:
            documents = self.load_directory(source)
        else:
            documents = self.load_single_document(source)

        # Split into chunks
        split_docs = self.split_documents(documents)

        logger.info(f"Pipeline complete: {len(split_docs)} chunks ready for embedding")

        return split_docs

    def get_document_stats(self, documents: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents.

        Args:
            documents: List of Document objects

        Returns:
            Dictionary with stats (total_docs, total_chars, avg_chunk_size, etc.)
        """
        if not documents:
            return {
                'total_docs': 0,
                'total_chars': 0,
                'avg_chunk_size': 0,
                'unique_sources': 0,
                'sources': []
            }

        # Calculate statistics
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars // len(documents) if documents else 0

        # Get unique source files
        unique_sources = set()
        for doc in documents:
            if 'source' in doc.metadata:
                unique_sources.add(doc.metadata['source'])

        return {
            'total_docs': len(documents),
            'total_chars': total_chars,
            'avg_chunk_size': avg_chunk_size,
            'unique_sources': len(unique_sources),
            'sources': list(unique_sources)
        }


class PDFLoader:
    """
    Specialized PDF loader with advanced options.
    Supports multiple PDF parsing backends for robustness.
    """

    def __init__(self, mode: str = "pypdf"):
        """
        Initialize PDF loader.

        Args:
            mode: PDF extraction mode ("pypdf", "pdfplumber")
        """
        self.mode = mode
        logger.info(f"PDFLoader initialized with mode: {mode}")

    def load(self, file_path: str, extract_images: bool = False) -> List[Any]:
        """
        Load PDF with specified backend.

        Args:
            file_path: Path to PDF file
            extract_images: Whether to extract images (requires OCR)

        Returns:
            List of Document objects
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Loading PDF with {self.mode} backend: {file_path}")

        try:
            # For now, use PyPDFLoader as it's most reliable
            # TODO: Future enhancement - support pdfplumber for better table extraction
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Add backend info to metadata
            for doc in documents:
                doc.metadata['pdf_backend'] = self.mode
                doc.metadata['extract_images'] = extract_images

            logger.info(f"Successfully loaded {len(documents)} pages from PDF")
            return documents

        except Exception as e:
            logger.error(f"Failed to load PDF: {str(e)}")
            raise RuntimeError(f"PDF loading failed: {str(e)}")
