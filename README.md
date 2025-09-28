Multi-Source FAQ Bot
A Python-based FAQ bot that leverages Retrieval-Augmented Generation (RAG) to answer questions using a small language model (Microsoft Phi-2) and information from multiple sources (PDFs, text files, and web pages). The bot retrieves relevant context using Sentence-Transformers and FAISS, then generates concise FAQ-style responses via a web interface powered by Gradio.
Features

Multi-Source Ingestion: Processes FAQs from PDFs, text files, and scraped web pages.
Efficient Retrieval: Uses Sentence-Transformers for embeddings and FAISS for fast similarity search.
Lightweight LLM: Employs Microsoft Phi-2 (~2.7B parameters) for low-resource compatibility.
Interactive UI: Gradio-based web interface for user queries with source citations.
Scalable Design: Easily extendable to additional data sources or larger models.

Requirements

Python 3.10+
~4GB RAM (CPU) or GPU (optional for faster inference)
Dependencies:pip install torch transformers sentence-transformers faiss-cpu gradio pypdf2 requests beautifulsoup4 accelerate



Setup

Clone the Repository:
git clone https://github.com/yourusername/multy-source-chat-bot.git
cd multy-source-chat-bot


Create a Virtual Environment:
python -m venv faq_bot_env
source faq_bot_env/bin/activate  # On Windows: faq_bot_env\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Prepare Sources:

Create a sources/ folder in the project root.
Add sample files (e.g., faq1.pdf, faq2.txt).
Optionally, update the web URL in ingest.py to a valid FAQ page (e.g., https://example.com/faq).


Index Sources:
python ingest.py

Generates faq_index.faiss and metadata.pkl for retrieval.

Run the Bot:
python app.py

Access the UI at http://127.0.0.1:7860 in your browser.


Usage

Open the Gradio interface in your browser.
Enter a question (e.g., "What is the refund policy?").
View the generated answer and cited sources.

Project Structure

ingest.py: Loads and indexes data from PDFs, text files, and web pages using Sentence-Transformers and FAISS.
retrieve.py: Retrieves top-k relevant chunks based on query embeddings.
generate.py: Uses Phi-2 to generate FAQ-style answers from retrieved context.
app.py: Launches the Gradio web interface for user interaction.
sources/: Directory for input files (PDFs, text files).
faq_index.faiss, metadata.pkl: Generated files for vector index and metadata.

Troubleshooting

Missing accelerate: Install with pip install accelerate.
Memory Issues: Use a smaller model like distilgpt2 in generate.py if Phi-2 crashes.
No Results: Ensure sources/ contains valid files and ingest.py has run successfully.
Web Scraping Errors: Verify the URL in ingest.py and check internet connectivity.

Future Improvements

Add support for CSVs or databases (e.g., SQLite).
Implement recursive text chunking with NLTK for better context.
Fine-tune Phi-2 for domain-specific FAQs.
Deploy to Hugging Face Spaces or FastAPI for production.

License
MIT License
