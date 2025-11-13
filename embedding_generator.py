"""
Embedding Generation Module

This module handles the generation of embeddings for document chunks
using LangChain's OpenAI embeddings.
"""

from typing import List
from langchain_openai import OpenAIEmbeddings
from pdf_processor import DocumentChunk
from error_handler import ErrorHandler, retry_with_exponential_backoff, log_function_call


class EmbeddingGenerator:
    """Generates embeddings for text using OpenAI's embedding models."""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the embedding model to use (default: text-embedding-ada-002)
        """
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model_name
        )
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    @log_function_call()
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of DocumentChunk objects to generate embeddings for
            
        Returns:
            List of DocumentChunk objects with embeddings populated
            
        Raises:
            Exception: If embedding generation fails
        """
        logger = ErrorHandler.get_logger()
        
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Extract text from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batch
            embeddings = self.embeddings.embed_documents(texts)
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    @log_function_call()
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for a query string.
        
        Args:
            query: Query text to generate embedding for
            
        Returns:
            Embedding vector as a list of floats
            
        Raises:
            Exception: If embedding generation fails
        """
        logger = ErrorHandler.get_logger()
        
        try:
            logger.debug(f"Generating query embedding for: {query[:50]}...")
            embedding = self.embeddings.embed_query(query)
            logger.debug(f"Generated query embedding with dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}", exc_info=True)
            raise Exception(f"Failed to generate query embedding: {str(e)}")
