# Quiz Generator

An AI-powered quiz generator that creates multiple-choice questions from PDF documents using LangChain, Pinecone, and OpenAI.

## Project Structure

```
quiz-generator/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and environment variable management
├── pdf_processor.py            # PDF text extraction and chunking
├── embedding_generator.py      # Embedding generation using LangChain
├── vector_store.py             # Pinecone vector store management
├── question_generator.py       # LLM-based question generation
├── error_handler.py            # Error handling utilities
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
└── .gitignore                 # Git ignore rules
```

## Setup Instructions

### 1. Clone the repository and navigate to the project directory

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., "us-west1-gcp")
- `PINECONE_INDEX_NAME`: Name for your Pinecone index (default: "quiz-generator")

### 5. Run the application

```bash
streamlit run app.py
```

## Environment Variables

Required:
- `OPENAI_API_KEY`: OpenAI API key for embeddings and question generation
- `PINECONE_API_KEY`: Pinecone API key for vector storage
- `PINECONE_ENVIRONMENT`: Pinecone environment identifier

Optional (with defaults):
- `PINECONE_INDEX_NAME`: Pinecone index name (default: "quiz-generator")
- `CHUNK_SIZE`: Text chunk size in characters (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks in characters (default: 200)
- `MAX_QUESTIONS`: Maximum number of questions allowed (default: 20)
- `MIN_QUESTIONS`: Minimum number of questions allowed (default: 1)

## Usage

1. Upload a PDF document through the web interface
2. Specify the number of quiz questions you want to generate (1-20)
3. Click "Generate Questions" to create the quiz
4. View the generated questions with multiple-choice options
5. Toggle to reveal correct answers

## Features

- PDF text extraction and processing
- Semantic search using vector embeddings
- AI-powered question generation
- Multiple-choice format with 4 options
- Answer reveal functionality
- Error handling and validation

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- Streamlit for web interface
- LangChain for AI orchestration
- Pinecone for vector storage
- OpenAI for embeddings and generation
- PyPDF2 for PDF processing
