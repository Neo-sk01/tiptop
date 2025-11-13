"""
Question Generation Module

This module handles AI-powered question generation from document content
using LangChain and OpenAI LLMs.
"""

from dataclasses import dataclass
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from vector_store import VectorStoreManager
from embedding_generator import EmbeddingGenerator
import re
from error_handler import ErrorHandler, retry_with_exponential_backoff, log_function_call


@dataclass
class Question:
    """Represents a generated quiz question with multiple choice options."""
    question_text: str
    options: List[str]  # Always 4 options
    correct_answer_index: int  # 0-3
    source_context: str
    id: Optional[str] = None


class QuestionGenerator:
    """Generates quiz questions from document content using LLM."""
    
    def __init__(
        self,
        llm_api_key: str,
        vector_store: VectorStoreManager,
        embedding_generator: EmbeddingGenerator,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize the question generator.
        
        Args:
            llm_api_key: OpenAI API key
            vector_store: VectorStoreManager instance for retrieving context
            embedding_generator: EmbeddingGenerator instance for query embeddings
            model_name: OpenAI model to use (default: gpt-3.5-turbo)
            temperature: LLM temperature for generation (default: 0.7)
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            api_key=llm_api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "num_questions"],
            template="""Given the following context from a document, generate {num_questions} multiple-choice questions.

Each question should:
- Test understanding of key concepts from the context
- Have exactly 4 options labeled A, B, C, D
- Have exactly one correct answer
- Be clear and unambiguous
- Be based directly on information in the context

Context:
{context}

Format each question EXACTLY as follows (include the separators):

---QUESTION---
Question: [question text]
A) [option 1]
B) [option 2]
C) [option 3]
D) [option 4]
Correct Answer: [A/B/C/D]
---END---

Generate {num_questions} questions now:"""
        )
    
    @retry_with_exponential_backoff(
        max_retries=3,
        initial_delay=2.0,
        exceptions=(Exception,)
    )
    @log_function_call()
    def generate_questions(
        self,
        num_questions: int,
        query: str = "Generate questions about the main topics",
        top_k: int = 5,
        namespace: str = ""
    ) -> List[Question]:
        """
        Generate quiz questions from document content.
        
        Args:
            num_questions: Number of questions to generate
            query: Query to retrieve relevant context (default: generic query)
            top_k: Number of document chunks to retrieve for context
            namespace: Pinecone namespace to search
            
        Returns:
            List of Question objects
            
        Raises:
            Exception: If question generation fails
        """
        logger = ErrorHandler.get_logger()
        
        try:
            logger.info(f"Generating {num_questions} questions from namespace '{namespace}'")
            
            # Generate query embedding
            logger.debug("Generating query embedding")
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            
            # Retrieve relevant context from vector store
            logger.debug(f"Retrieving top {top_k} relevant chunks")
            relevant_chunks = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=top_k,
                namespace=namespace
            )
            
            if not relevant_chunks:
                logger.error("No relevant content found in vector store")
                raise Exception("No relevant content found in vector store")
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Combine chunk texts for context
            context = "\n\n".join([chunk.text for chunk in relevant_chunks])
            logger.debug(f"Combined context length: {len(context)} characters")
            
            # Generate prompt
            prompt = self.prompt_template.format(
                context=context,
                num_questions=num_questions
            )
            
            # Call LLM
            logger.info("Calling LLM to generate questions")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw_output = response.content
            
            logger.debug(f"LLM response length: {len(raw_output)} characters")
            
            # Parse questions from output
            questions = self._parse_questions(raw_output, context)
            logger.info(f"Parsed {len(questions)} questions from LLM output")
            
            # Validate and filter questions
            valid_questions = []
            for question in questions:
                if self.validate_question(question):
                    valid_questions.append(question)
                else:
                    logger.warning(f"Question failed validation: {question.question_text[:50]}...")
            
            logger.info(f"Validated {len(valid_questions)} questions")
            
            # If we don't have enough valid questions, raise an error
            if len(valid_questions) < num_questions:
                logger.warning(
                    f"Only generated {len(valid_questions)} valid questions out of {num_questions} requested"
                )
            
            return valid_questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Failed to generate questions: {str(e)}", exc_info=True)
            raise Exception(f"Failed to generate questions: {str(e)}")
    
    def _parse_questions(self, raw_output: str, source_context: str) -> List[Question]:
        """
        Parse LLM output into Question objects.
        
        Args:
            raw_output: Raw text output from LLM
            source_context: Source context used for generation
            
        Returns:
            List of Question objects
        """
        questions = []
        
        # Split by question separator
        question_blocks = raw_output.split("---QUESTION---")
        
        for block in question_blocks:
            if "---END---" not in block:
                continue
            
            # Extract content between markers
            content = block.split("---END---")[0].strip()
            
            try:
                question = self.format_question(content, source_context)
                if question:
                    questions.append(question)
            except Exception as e:
                print(f"Warning: Failed to parse question block: {e}")
                continue
        
        return questions
    
    def format_question(self, raw_text: str, source_context: str) -> Optional[Question]:
        """
        Parse raw question text into a Question object.
        
        Args:
            raw_text: Raw question text from LLM
            source_context: Source context for the question
            
        Returns:
            Question object or None if parsing fails
        """
        try:
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            
            # Extract question text
            question_text = None
            for line in lines:
                if line.startswith("Question:"):
                    question_text = line.replace("Question:", "").strip()
                    break
            
            if not question_text:
                return None
            
            # Extract options
            options = []
            option_pattern = re.compile(r'^([A-D])\)\s*(.+)$')
            
            for line in lines:
                match = option_pattern.match(line)
                if match:
                    options.append(match.group(2).strip())
            
            if len(options) != 4:
                return None
            
            # Extract correct answer
            correct_answer_index = None
            for line in lines:
                if line.startswith("Correct Answer:"):
                    answer_letter = line.replace("Correct Answer:", "").strip().upper()
                    if answer_letter in ['A', 'B', 'C', 'D']:
                        correct_answer_index = ord(answer_letter) - ord('A')
                    break
            
            if correct_answer_index is None:
                return None
            
            # Create Question object
            return Question(
                question_text=question_text,
                options=options,
                correct_answer_index=correct_answer_index,
                source_context=source_context[:500]  # Truncate context for storage
            )
            
        except Exception as e:
            print(f"Error formatting question: {e}")
            return None
    
    def validate_question(self, question: Question) -> bool:
        """
        Validate that a question meets quality requirements.
        
        Args:
            question: Question object to validate
            
        Returns:
            bool: True if question is valid, False otherwise
        """
        # Check question text is not empty
        if not question.question_text or len(question.question_text.strip()) == 0:
            return False
        
        # Check exactly 4 options
        if len(question.options) != 4:
            return False
        
        # Check all options are non-empty
        for option in question.options:
            if not option or len(option.strip()) == 0:
                return False
        
        # Check correct answer index is valid (0-3)
        if question.correct_answer_index not in [0, 1, 2, 3]:
            return False
        
        # Check options are unique
        if len(set(question.options)) != 4:
            return False
        
        return True
