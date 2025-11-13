"""
PDF Processing Module

This module handles PDF text extraction, chunking, and metadata extraction
for the Quiz Generator application.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import PyPDF2
from io import BytesIO
from error_handler import ErrorHandler, log_function_call


@dataclass
class DocumentChunk:
    """Represents a text chunk extracted from a PDF document with metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class PDFProcessor:
    """Handles PDF document processing including text extraction and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks in characters (default: 1000)
            chunk_overlap: Number of overlapping characters between chunks (default: 200)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @log_function_call()
    def extract_text(self, pdf_file) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_file: Uploaded PDF file (Streamlit UploadedFile or file-like object)
            
        Returns:
            Extracted text as a single string
            
        Raises:
            ValueError: If the PDF is invalid or cannot be read
            Exception: For other PDF processing errors
        """
        logger = ErrorHandler.get_logger()
        
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Check if PDF has pages
            if len(pdf_reader.pages) == 0:
                logger.error("PDF file contains no pages")
                raise ValueError("PDF file contains no pages")
            
            logger.info(f"Extracting text from PDF with {len(pdf_reader.pages)} pages")
            
            # Extract text from all pages
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            
            # Combine all text
            full_text = "\n".join(text_content)
            
            if not full_text.strip():
                logger.error("No text content could be extracted from the PDF")
                raise ValueError("No text content could be extracted from the PDF")
            
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"PDF read error: {str(e)}", exc_info=True)
            raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            raise Exception(f"Error processing PDF: {str(e)}")

    @log_function_call()
    def chunk_text(self, text: str, document_name: str = "document") -> List[DocumentChunk]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: The text to chunk
            document_name: Name of the source document for metadata
            
        Returns:
            List of DocumentChunk objects with text and metadata
        """
        logger = ErrorHandler.get_logger()
        
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        logger.info(f"Chunking text of length {len(text)} with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position for this chunk
            end = start + self.chunk_size
            
            # Extract chunk text
            chunk_text = text[start:end]
            
            # Create DocumentChunk with metadata
            chunk = DocumentChunk(
                id=f"{document_name}_chunk_{chunk_index}",
                text=chunk_text.strip(),
                metadata={
                    "document_name": document_name,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": min(end, len(text)),
                    "chunk_size": len(chunk_text.strip())
                }
            )
            
            chunks.append(chunk)
            
            # Move start position forward, accounting for overlap
            start = end - self.chunk_overlap
            chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def extract_metadata(self, pdf_file) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_file: Uploaded PDF file (Streamlit UploadedFile or file-like object)
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract basic metadata
            metadata = {
                "num_pages": len(pdf_reader.pages),
                "file_name": getattr(pdf_file, 'name', 'unknown.pdf')
            }
            
            # Extract PDF metadata if available
            if pdf_reader.metadata:
                pdf_meta = pdf_reader.metadata
                metadata.update({
                    "title": pdf_meta.get('/Title', ''),
                    "author": pdf_meta.get('/Author', ''),
                    "subject": pdf_meta.get('/Subject', ''),
                    "creator": pdf_meta.get('/Creator', ''),
                    "producer": pdf_meta.get('/Producer', ''),
                    "creation_date": pdf_meta.get('/CreationDate', ''),
                })
            
            return metadata
            
        except Exception as e:
            # Return minimal metadata on error
            return {
                "num_pages": 0,
                "file_name": getattr(pdf_file, 'name', 'unknown.pdf'),
                "error": str(e)
            }
