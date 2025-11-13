"""
Error Handler Module

This module provides centralized error handling, logging, and retry logic
for the Quiz Generator application.
"""

import logging
import time
import traceback
from typing import Tuple, Optional, Callable, Any
from functools import wraps
from enum import Enum


class ErrorType(Enum):
    """Enumeration of error types in the application."""
    PDF_ERROR = "pdf_error"
    VECTOR_STORE_ERROR = "vector_store_error"
    GENERATION_ERROR = "generation_error"
    CONFIGURATION_ERROR = "configuration_error"
    VALIDATION_ERROR = "validation_error"
    GENERIC_ERROR = "generic_error"


class ErrorHandler:
    """Centralized error handling and logging for the application."""
    
    _logger: Optional[logging.Logger] = None
    
    @staticmethod
    def setup_logging(
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        log_format: Optional[str] = None
    ) -> None:
        """
        Configure application logging.
        
        Args:
            log_level: Logging level (default: INFO)
            log_file: Optional log file path
            log_format: Optional custom log format
        """
        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[]
        )
        
        # Create logger
        logger = logging.getLogger('quiz_generator')
        logger.setLevel(log_level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        ErrorHandler._logger = logger
        logger.info("Logging configured successfully")
    
    @staticmethod
    def get_logger() -> logging.Logger:
        """
        Get the application logger.
        
        Returns:
            Logger instance
        """
        if ErrorHandler._logger is None:
            ErrorHandler.setup_logging()
        return ErrorHandler._logger
    
    @staticmethod
    def handle_pdf_error(error: Exception) -> Tuple[str, str]:
        """
        Handle PDF processing errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Tuple of (user_friendly_message, technical_details)
        """
        logger = ErrorHandler.get_logger()
        logger.error(f"PDF processing error: {str(error)}", exc_info=True)
        
        error_str = str(error).lower()
        
        # Determine specific error type
        if "corrupted" in error_str or "invalid" in error_str:
            user_msg = "The PDF file appears to be corrupted or invalid. Please try a different PDF file."
        elif "no pages" in error_str or "empty" in error_str:
            user_msg = "The PDF file is empty or contains no readable content. Please upload a PDF with text content."
        elif "no text" in error_str or "extract" in error_str:
            user_msg = "Unable to extract text from the PDF. The file may contain only images or scanned content."
        elif "password" in error_str or "encrypted" in error_str:
            user_msg = "The PDF file is password-protected or encrypted. Please upload an unprotected PDF."
        else:
            user_msg = "Failed to process the PDF file. Please ensure it's a valid PDF document."
        
        technical_details = f"Error Type: PDF Processing Error\n{traceback.format_exc()}"
        
        return user_msg, technical_details
    
    @staticmethod
    def handle_vector_store_error(error: Exception) -> Tuple[str, str]:
        """
        Handle vector store (Pinecone) errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Tuple of (user_friendly_message, technical_details)
        """
        logger = ErrorHandler.get_logger()
        logger.error(f"Vector store error: {str(error)}", exc_info=True)
        
        error_str = str(error).lower()
        
        # Determine specific error type
        if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            user_msg = "Vector database authentication failed. Please check your Pinecone API key configuration."
        elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
            user_msg = "Unable to connect to the vector database. Please check your internet connection and try again."
        elif "index" in error_str and "not found" in error_str:
            user_msg = "Vector database index not found. The system will attempt to create it automatically."
        elif "quota" in error_str or "limit" in error_str:
            user_msg = "Vector database quota exceeded. Please check your Pinecone plan limits."
        elif "dimension" in error_str:
            user_msg = "Vector dimension mismatch. Please contact support."
        else:
            user_msg = "Failed to store or retrieve data from the vector database. Please try again."
        
        technical_details = f"Error Type: Vector Store Error\n{traceback.format_exc()}"
        
        return user_msg, technical_details
    
    @staticmethod
    def handle_generation_error(error: Exception) -> Tuple[str, str]:
        """
        Handle question generation (LLM) errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Tuple of (user_friendly_message, technical_details)
        """
        logger = ErrorHandler.get_logger()
        logger.error(f"Question generation error: {str(error)}", exc_info=True)
        
        error_str = str(error).lower()
        
        # Determine specific error type
        if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            user_msg = "OpenAI API authentication failed. Please check your API key configuration."
        elif "rate limit" in error_str or "quota" in error_str:
            user_msg = "API rate limit exceeded. Please wait a moment and try again."
        elif "timeout" in error_str or "connection" in error_str:
            user_msg = "Connection to OpenAI API timed out. Please check your internet connection and try again."
        elif "no relevant content" in error_str:
            user_msg = "No relevant content found in the document for question generation. Please try a different PDF."
        elif "parse" in error_str or "format" in error_str:
            user_msg = "Failed to parse generated questions. Please try again."
        elif "model" in error_str:
            user_msg = "The specified AI model is not available. Please contact support."
        else:
            user_msg = "Failed to generate questions. Please try again."
        
        technical_details = f"Error Type: Question Generation Error\n{traceback.format_exc()}"
        
        return user_msg, technical_details
    
    @staticmethod
    def handle_configuration_error(error: Exception) -> Tuple[str, str]:
        """
        Handle configuration errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Tuple of (user_friendly_message, technical_details)
        """
        logger = ErrorHandler.get_logger()
        logger.error(f"Configuration error: {str(error)}", exc_info=True)
        
        error_str = str(error).lower()
        
        # Determine specific error type
        if "api key" in error_str or "openai" in error_str:
            user_msg = "OpenAI API key is missing or invalid. Please set OPENAI_API_KEY in your .env file."
        elif "pinecone" in error_str:
            user_msg = "Pinecone configuration is missing or invalid. Please check your Pinecone settings in .env file."
        elif "environment variable" in error_str:
            user_msg = "Required environment variables are missing. Please check your .env file configuration."
        else:
            user_msg = f"Configuration error: {str(error)}"
        
        technical_details = f"Error Type: Configuration Error\n{traceback.format_exc()}"
        
        return user_msg, technical_details
    
    @staticmethod
    def handle_validation_error(error: Exception) -> Tuple[str, str]:
        """
        Handle validation errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Tuple of (user_friendly_message, technical_details)
        """
        logger = ErrorHandler.get_logger()
        logger.warning(f"Validation error: {str(error)}")
        
        user_msg = f"Validation error: {str(error)}"
        technical_details = f"Error Type: Validation Error\n{str(error)}"
        
        return user_msg, technical_details
    
    @staticmethod
    def handle_generic_error(error: Exception) -> Tuple[str, str]:
        """
        Handle generic/unknown errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Tuple of (user_friendly_message, technical_details)
        """
        logger = ErrorHandler.get_logger()
        logger.error(f"Unexpected error: {str(error)}", exc_info=True)
        
        user_msg = "An unexpected error occurred. Please try again or contact support if the problem persists."
        technical_details = f"Error Type: Generic Error\n{traceback.format_exc()}"
        
        return user_msg, technical_details


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[type, ...] = (Exception,)
) -> Callable:
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff calculation
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = ErrorHandler.get_logger()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {str(e)}",
                            exc_info=True
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.2f} seconds... Error: {str(e)}"
                    )
                    
                    time.sleep(delay)
            
            return None
        
        return wrapper
    return decorator


def log_function_call(log_args: bool = False, log_result: bool = False) -> Callable:
    """
    Decorator to log function calls.
    
    Args:
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        
    Returns:
        Decorated function with logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = ErrorHandler.get_logger()
            
            func_name = func.__name__
            
            # Log function entry
            if log_args:
                logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            else:
                logger.debug(f"Calling {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                # Log function exit
                if log_result:
                    logger.debug(f"{func_name} returned: {result}")
                else:
                    logger.debug(f"{func_name} completed successfully")
                
                return result
            except Exception as e:
                logger.error(f"{func_name} raised exception: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    return decorator
