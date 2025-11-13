"""
Quiz Generator - Streamlit Application

Main application entry point for the PDF Quiz Generator.
Provides a web interface for uploading PDFs and generating quiz questions.
"""

import streamlit as st
from typing import Optional, List
import traceback

# Import application modules
from config import load_config, ConfigurationError
from pdf_processor import PDFProcessor, DocumentChunk
from vector_store import VectorStoreManager
from embedding_generator import EmbeddingGenerator
from question_generator import QuestionGenerator, Question


# Page configuration
st.set_page_config(
    page_title="Quiz Generator",
    page_icon="📝",
    layout="wide"
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = None
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = None
    if 'show_answers' not in st.session_state:
        st.session_state.show_answers = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False


def display_error(message: str, details: Optional[str] = None):
    """
    Display an error message to the user.
    
    Args:
        message: Main error message
        details: Optional detailed error information
    """
    st.error(f"❌ {message}")
    if details:
        with st.expander("Error Details"):
            st.code(details)


def display_success(message: str):
    """
    Display a success message to the user.
    
    Args:
        message: Success message
    """
    st.success(f"✅ {message}")


def display_questions(questions: List[Question]):
    """
    Display generated questions with multiple choice options and answer reveal.
    
    Args:
        questions: List of Question objects to display
    """
    st.divider()
    st.subheader("📋 Generated Questions")
    
    # Toggle for showing/hiding answers
    col1, col2 = st.columns([3, 1])
    with col2:
        show_answers = st.checkbox(
            "Show Answers",
            value=st.session_state.show_answers,
            key="show_answers_toggle"
        )
        st.session_state.show_answers = show_answers
    
    # Display each question
    for idx, question in enumerate(questions, 1):
        # Create a container for each question with styling
        with st.container():
            # Question header with number
            st.markdown(f"### Question {idx}")
            
            # Question text
            st.markdown(f"**{question.question_text}**")
            
            # Display options with letter labels
            option_letters = ['A', 'B', 'C', 'D']
            
            for i, option in enumerate(question.options):
                option_letter = option_letters[i]
                
                # Check if this is the correct answer
                is_correct = (i == question.correct_answer_index)
                
                # Style the option based on whether answers are shown
                if show_answers and is_correct:
                    # Highlight correct answer with green background
                    st.markdown(
                        f"""<div style="background-color: #d4edda; 
                        border-left: 4px solid #28a745; 
                        padding: 10px; 
                        margin: 5px 0; 
                        border-radius: 4px;">
                        <strong style="color: #155724;">✓ {option_letter}) {option}</strong>
                        </div>""",
                        unsafe_allow_html=True
                    )
                else:
                    # Regular option display
                    st.markdown(f"{option_letter}) {option}")
            
            # Show correct answer indicator when answers are revealed
            if show_answers:
                correct_letter = option_letters[question.correct_answer_index]
                st.success(f"✅ Correct Answer: {correct_letter}")
            
            # Add spacing between questions
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary at the bottom
    st.divider()
    st.info(f"📊 Total Questions: {len(questions)}")


def render_header():
    """Render the application header."""
    st.title("📝 Quiz Generator")
    st.markdown("""
    Upload a PDF document and automatically generate quiz questions from its content.
    The system uses AI to create relevant multiple-choice questions based on the document.
    """)
    st.divider()


def render_upload_section():
    """
    Render the file upload section.
    
    Returns:
        Uploaded file object or None
    """
    st.subheader("1. Upload PDF Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to generate quiz questions from",
        key="pdf_uploader"
    )
    
    return uploaded_file


def render_question_count_input(min_val: int = 1, max_val: int = 20):
    """
    Render the question count input widget.
    
    Args:
        min_val: Minimum number of questions
        max_val: Maximum number of questions
        
    Returns:
        Number of questions to generate
    """
    st.subheader("2. Configure Quiz")
    
    num_questions = st.number_input(
        "Number of questions to generate",
        min_value=min_val,
        max_value=max_val,
        value=5,
        step=1,
        help=f"Choose between {min_val} and {max_val} questions"
    )
    
    return int(num_questions)


def render_generate_button():
    """
    Render the generate questions button.
    
    Returns:
        bool: True if button was clicked
    """
    st.subheader("3. Generate Questions")
    
    button_clicked = st.button(
        "🚀 Generate Questions",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.uploaded_file is None
    )
    
    if st.session_state.uploaded_file is None:
        st.info("👆 Please upload a PDF file first")
    
    return button_clicked


def validate_file_upload(uploaded_file) -> bool:
    """
    Validate the uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        bool: True if file is valid
    """
    if uploaded_file is None:
        display_error("No file uploaded", "Please upload a PDF file to continue.")
        return False
    
    # Check file extension
    if not uploaded_file.name.lower().endswith('.pdf'):
        display_error(
            "Invalid file format",
            f"File '{uploaded_file.name}' is not a PDF. Please upload a PDF file."
        )
        return False
    
    # Check file size (limit to 50MB)
    max_size = 50 * 1024 * 1024  # 50MB in bytes
    if uploaded_file.size > max_size:
        display_error(
            "File too large",
            f"File size is {uploaded_file.size / (1024*1024):.1f}MB. Maximum allowed size is 50MB."
        )
        return False
    
    return True


def validate_question_count(num_questions: int, min_val: int, max_val: int) -> bool:
    """
    Validate the question count input.
    
    Args:
        num_questions: Number of questions requested
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        bool: True if valid
    """
    if num_questions < min_val or num_questions > max_val:
        display_error(
            "Invalid question count",
            f"Please enter a number between {min_val} and {max_val}."
        )
        return False
    
    return True


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Load configuration
    try:
        if st.session_state.config is None:
            with st.spinner("Loading configuration..."):
                st.session_state.config = load_config()
    except ConfigurationError as e:
        display_error(
            "Configuration Error",
            f"{str(e)}\n\nPlease ensure all required environment variables are set in your .env file."
        )
        st.stop()
    except Exception as e:
        display_error(
            "Unexpected configuration error",
            f"{str(e)}\n\n{traceback.format_exc()}"
        )
        st.stop()
    
    config = st.session_state.config
    
    # Render upload section
    uploaded_file = render_upload_section()
    
    # Update session state if file changed
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.processed_chunks = None
        st.session_state.generated_questions = None
        st.session_state.processing_complete = False
        st.session_state.error_message = None
    
    # Render question count input
    num_questions = render_question_count_input(
        min_val=config.min_questions,
        max_val=config.max_questions
    )
    
    # Render generate button
    generate_clicked = render_generate_button()
    
    # Handle generate button click
    if generate_clicked:
        # Clear previous results and errors
        st.session_state.generated_questions = None
        st.session_state.error_message = None
        st.session_state.processing_complete = False
        
        # Validate inputs
        if not validate_file_upload(uploaded_file):
            st.stop()
        
        if not validate_question_count(num_questions, config.min_questions, config.max_questions):
            st.stop()
        
        # Process PDF and generate questions
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Process PDF
            status_text.text("📄 Processing PDF document...")
            progress_bar.progress(20)
            
            pdf_processor = PDFProcessor(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            
            # Extract text
            text = pdf_processor.extract_text(uploaded_file)
            
            # Extract metadata
            metadata = pdf_processor.extract_metadata(uploaded_file)
            
            # Chunk text
            chunks = pdf_processor.chunk_text(text, metadata.get('file_name', 'document'))
            st.session_state.processed_chunks = chunks
            
            display_success(f"Processed PDF: {len(chunks)} text chunks created")
            progress_bar.progress(40)
            
            # Step 2: Generate embeddings and store in vector database
            status_text.text("🔢 Generating embeddings...")
            
            embedding_generator = EmbeddingGenerator(api_key=config.openai_api_key)
            
            # Generate embeddings for all chunks
            chunks = embedding_generator.generate_embeddings(chunks)
            
            progress_bar.progress(60)
            
            # Step 3: Store in Pinecone
            status_text.text("💾 Storing in vector database...")
            
            vector_store = VectorStoreManager(
                api_key=config.pinecone_api_key,
                environment=config.pinecone_environment,
                index_name=config.pinecone_index_name
            )
            
            vector_store.initialize_index()
            
            # Use file name as namespace for isolation
            namespace = uploaded_file.name.replace('.pdf', '').replace(' ', '_')
            vector_store.store_embeddings(chunks, namespace=namespace)
            
            display_success("Document stored in vector database")
            progress_bar.progress(80)
            
            # Step 4: Generate questions
            status_text.text("🤖 Generating quiz questions...")
            
            question_generator = QuestionGenerator(
                llm_api_key=config.openai_api_key,
                vector_store=vector_store,
                embedding_generator=embedding_generator
            )
            
            questions = question_generator.generate_questions(
                num_questions=num_questions,
                namespace=namespace
            )
            
            st.session_state.generated_questions = questions
            st.session_state.processing_complete = True
            
            progress_bar.progress(100)
            status_text.text("✨ Questions generated successfully!")
            
            display_success(f"Generated {len(questions)} quiz questions!")
            
        except ValueError as e:
            display_error("PDF Processing Error", str(e))
            st.session_state.error_message = str(e)
        except ConfigurationError as e:
            display_error("Configuration Error", str(e))
            st.session_state.error_message = str(e)
        except Exception as e:
            display_error(
                "An unexpected error occurred",
                f"{str(e)}\n\n{traceback.format_exc()}"
            )
            st.session_state.error_message = str(e)
    
    # Display generated questions
    if st.session_state.generated_questions:
        display_questions(st.session_state.generated_questions)


if __name__ == "__main__":
    main()
