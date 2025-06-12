import json
import os
from typing import Dict, Any, List, Optional
import logging

class DocumentLoader:
    """Loads and manages policy documents from JSON files."""
    
    def __init__(self, data_paths: Dict[str, str]):
        """
        Initialize the document loader.
        
        Args:
            data_paths: Dictionary mapping document types to file paths
                       e.g., {"company_policies": "path/to/satim_chunks_cleaned.json",
                              "reference_policies": "path/to/pci_dss_chunks.json"}
        """
        self.logger = logging.getLogger("document_loader")
        self.data_paths = data_paths
        self.documents = {}
        self.document_chunks = {}
        self._load_all_documents()
    
    def _load_all_documents(self):
        """Load all documents from the specified paths."""
        for doc_type, file_path in self.data_paths.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    
                    self.document_chunks[doc_type] = chunks
                    self.logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
                    
                    # Group chunks by document for easier retrieval
                    doc_groups = {}
                    for chunk in chunks:
                        doc_name = chunk.get('document', 'unknown')
                        if doc_name not in doc_groups:
                            doc_groups[doc_name] = []
                        doc_groups[doc_name].append(chunk)
                    
                    self.documents[doc_type] = doc_groups
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path}: {e}")
            else:
                self.logger.warning(f"File not found: {file_path}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier (can be document name or type)
            
        Returns:
            Document data with combined content from all chunks
        """
        # Search through all document types
        for doc_type, doc_groups in self.documents.items():
            for doc_name, chunks in doc_groups.items():
                if doc_id in doc_name.lower() or doc_name.lower() in doc_id.lower():
                    # Combine all chunks for this document
                    combined_content = []
                    for chunk in chunks:
                        section_title = chunk.get('section_title', '')
                        text = chunk.get('text', '')
                        if section_title:
                            combined_content.append(f"## {section_title}\n{text}")
                        else:
                            combined_content.append(text)
                    
                    return {
                        "title": doc_name,
                        "content": "\n\n".join(combined_content),
                        "metadata": {
                            "document_type": doc_type,
                            "chunk_count": len(chunks),
                            "source": "processed_json"
                        }
                    }
        return None
    
    def search_documents(self, query: str, doc_type: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks based on query.
        
        Args:
            query: Search query
            doc_type: Specific document type to search in (optional)
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks
        """
        query_lower = query.lower()
        results = []
        
        # Determine which document types to search
        search_types = [doc_type] if doc_type else list(self.document_chunks.keys())
        
        for doc_type in search_types:
            if doc_type not in self.document_chunks:
                continue
                
            chunks = self.document_chunks[doc_type]
            for chunk in chunks:
                text = chunk.get('text', '').lower()
                section_title = chunk.get('section_title', '').lower()
                
                # Simple keyword matching (can be enhanced with embeddings)
                score = 0
                query_terms = query_lower.split()
                
                for term in query_terms:
                    if term in text:
                        score += text.count(term) * 1
                    if term in section_title:
                        score += section_title.count(term) * 2  # Higher weight for title matches
                
                if score > 0:
                    results.append({
                        "chunk": chunk,
                        "score": score,
                        "document_type": doc_type
                    })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_all_documents(self, doc_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all documents, optionally filtered by type.
        
        Args:
            doc_type: Optional document type filter
            
        Returns:
            Dictionary of documents
        """
        if doc_type:
            return self.documents.get(doc_type, {})
        return self.documents
    
    def get_document_types(self) -> List[str]:
        """Get all available document types."""
        return list(self.documents.keys())