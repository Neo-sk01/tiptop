"""
Configuration module for Quiz Generator application.
Loads and validates environment variables and application settings.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class AppConfig:
    """Application configuration settings."""
    openai_api_key: str
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_questions: int = 20
    min_questions: int = 1


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def load_config() -> AppConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        AppConfig: Application configuration object
        
    Raises:
        ConfigurationError: If required environment variables are missing or invalid
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Required environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'quiz-generator')
    
    # Validate required variables
    missing_vars = []
    if not openai_api_key:
        missing_vars.append('OPENAI_API_KEY')
    if not pinecone_api_key:
        missing_vars.append('PINECONE_API_KEY')
    if not pinecone_environment:
        missing_vars.append('PINECONE_ENVIRONMENT')
    
    if missing_vars:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            f"Please check your .env file or environment configuration."
        )
    
    # Optional settings with defaults
    try:
        chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        max_questions = int(os.getenv('MAX_QUESTIONS', '20'))
        min_questions = int(os.getenv('MIN_QUESTIONS', '1'))
    except ValueError as e:
        raise ConfigurationError(f"Invalid numeric configuration value: {e}")
    
    # Validate settings
    if chunk_size <= 0:
        raise ConfigurationError("CHUNK_SIZE must be greater than 0")
    if chunk_overlap < 0:
        raise ConfigurationError("CHUNK_OVERLAP must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ConfigurationError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    if min_questions < 1:
        raise ConfigurationError("MIN_QUESTIONS must be at least 1")
    if max_questions < min_questions:
        raise ConfigurationError("MAX_QUESTIONS must be greater than or equal to MIN_QUESTIONS")
    
    return AppConfig(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        pinecone_index_name=pinecone_index_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_questions=max_questions,
        min_questions=min_questions
    )


def validate_api_keys(config: AppConfig) -> bool:
    """
    Validate that API keys are present and non-empty.
    
    Args:
        config: Application configuration object
        
    Returns:
        bool: True if all API keys are valid
    """
    return bool(config.openai_api_key and config.pinecone_api_key)
