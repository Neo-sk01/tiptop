"""
Vector Store Module

This module manages the Pinecone vector store for storing and retrieving
document embeddings for semantic search.
"""

from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from pdf_processor import DocumentChunk
import time


class VectorStoreManager:
    """Manages Pinecone vector store operations for document embeddings."""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize the vector store manager.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-east-1-aws')
            index_name: Name of the Pinecone index to use
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)
        self.index = None
    
    def initialize_index(self, dimension: int = 1536, metric: str = "cosine") -> None:
        """
        Create or connect to a Pinecone index.
        
        Args:
            dimension: Dimension of the embedding vectors (default: 1536 for OpenAI ada-002)
            metric: Distance metric to use (default: cosine)
            
        Raises:
            Exception: If index creation or connection fails
        """
        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                # Create new index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.environment
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone index: {str(e)}")
    
    def store_embeddings(self, chunks: List[DocumentChunk], namespace: str = "") -> None:
        """
        Store document chunks with embeddings in Pinecone.
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
            namespace: Optional namespace for organizing vectors
            
        Raises:
            Exception: If storage operation fails
        """
        if not self.index:
            raise Exception("Index not initialized. Call initialize_index() first.")
        
        if not chunks:
            return
        
        try:
            # Prepare vectors for upsert
            vectors = []
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} has no embedding")
                
                vectors.append({
                    "id": chunk.id,
                    "values": chunk.embedding,
                    "metadata": {
                        "text": chunk.text,
                        **chunk.metadata
                    }
                })
            
            # Upsert vectors in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
            
        except Exception as e:
            raise Exception(f"Failed to store embeddings in Pinecone: {str(e)}")
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant chunks based on similarity to query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return (default: 5)
            namespace: Optional namespace to search within
            filter_dict: Optional metadata filter
            
        Returns:
            List of DocumentChunk objects matching the query
            
        Raises:
            Exception: If search operation fails
        """
        if not self.index:
            raise Exception("Index not initialized. Call initialize_index() first.")
        
        try:
            # Query the index
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                namespace=namespace,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Convert results to DocumentChunk objects
            chunks = []
            for match in results.matches:
                metadata = match.metadata.copy()
                text = metadata.pop("text", "")
                
                chunk = DocumentChunk(
                    id=match.id,
                    text=text,
                    metadata=metadata,
                    embedding=match.values if hasattr(match, 'values') else None
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Failed to perform similarity search: {str(e)}")
    
    def clear_index(self, namespace: str = "") -> None:
        """
        Clear all vectors from the index.
        
        Args:
            namespace: Optional namespace to clear (empty string clears default namespace)
            
        Raises:
            Exception: If clear operation fails
        """
        if not self.index:
            raise Exception("Index not initialized. Call initialize_index() first.")
        
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            raise Exception(f"Failed to clear index: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary containing index statistics
            
        Raises:
            Exception: If stats retrieval fails
        """
        if not self.index:
            raise Exception("Index not initialized. Call initialize_index() first.")
        
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            raise Exception(f"Failed to get index stats: {str(e)}")
