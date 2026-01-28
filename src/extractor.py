import json
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from src.config import AppConfig
from src.logger import logger

class MetadataExtractor:
    """
    Extracts structured metadata from document text using a Local LLM.
    
    Features:
    - Extracts: Title, Summary, Department, Date, Tags
    - JSON Output enforcement
    - Fallback to basic metadata on failure
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.enabled = config.ingestion.extraction.enabled
        
        if self.enabled:
            logger.info(f"Metadata Extraction Enabled. Model: {config.ingestion.extraction.model}")
            self.llm = Ollama(
                model=config.ingestion.extraction.model,
                base_url=config.llm.base_url,
                temperature=0.0, # Deterministic for extraction
                format="json"    # Enforce JSON mode in Ollama
            )
            self.prompt = self._build_prompt()
            
    def _build_prompt(self) -> PromptTemplate:
        template = """
        Analyze the following document text and extract structured metadata.
        Return ONLY a legitimate JSON object. Do not add any markdown formatting or explanation.

        Document Preview:
        {text}

        Extract the following fields:
        - title: A concise, descriptive title.
        - summary: A one-sentence summary.
        - department: Infer the department (e.g., Engineering, HR, Academic, Finance, Legal). If unknown, use "General".
        - date: The primary date mentioned (YYYY-MM-DD or YYYY). If unknown, use null.
        - tags: A list of 3-5 relevant keywords.

        JSON Structure:
        {{
            "title": "string",
            "summary": "string",
            "department": "string",
            "date": "string",
            "tags": ["string", "string"]
        }}
        """
        return PromptTemplate(template=template, input_variables=["text"])

    def extract(self, text: str, file_name: str) -> Dict[str, Any]:
        """
        Extract metadata from the first chunk of text.
        Returns a dict of metadata.
        """
        if not self.enabled:
            return self._basic_fallback(file_name)

        try:
            # We only need the first ~2k chars to get the gist
            formatted_prompt = self.prompt.format(text=text[:2000])
            
            logger.debug(f"Extracting metadata for {file_name}...")
            response = self.llm.invoke(formatted_prompt)
            
            # Parse JSON
            metadata = json.loads(response)
            
            # Ensure all keys exist
            defaults = {
                "title": file_name,
                "summary": "No summary available.",
                "department": "General",
                "date": None,
                "tags": []
            }
            # Merge defaults with result
            final_metadata = {**defaults, **metadata}
            
            logger.debug(f"Extracted: {final_metadata['title']} ({final_metadata['department']})")
            return final_metadata

        except Exception as e:
            logger.warning(f"Metadata extraction failed for {file_name}: {e}. Using fallback.")
            return self._basic_fallback(file_name)

    def _basic_fallback(self, file_name: str) -> Dict[str, Any]:
        """Fallback metadata if LLM fails or is disabled."""
        return {
            "title": file_name,
            "summary": "Metadata extraction disabled or failed.",
            "department": "Uncategorized",
            "date": None,
            "tags": []
        }
