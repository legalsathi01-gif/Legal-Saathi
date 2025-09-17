# backend/utils.py
import fitz  # PyMuPDF
from typing import List, Optional

# New imports for semantic search
from sentence_transformers import SentenceTransformer, util


# --- PDF and Text Processing Functions ---

def extract_text_from_pdf(pdf_path: Optional[str] = None, stream: Optional[bytes] = None) -> List[str]:
    """
    Extracts text from a PDF, returning a list of strings where each string is the text of a page.
    Can load from a file path or an in-memory byte stream.
    """
    try:
        # Open the PDF from a stream (for uploads) or a file path
        doc = fitz.open(stream=stream, filetype="pdf") if stream else fitz.open(pdf_path)
        
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        return pages_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return []


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Splits a long text into smaller, overlapping chunks."""
    if not isinstance(text, str) or not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("Overlap size must be smaller than chunk size.")
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        start_index += chunk_size - overlap
    return chunks


# --- Retrieval Functions (NEW CODE GOES HERE) ---

# Option 1: Simple keyword search (fast, no extra deps)
def find_relevant_chunks_keyword(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Finds the most relevant chunks based on keyword frequency."""
    query_words = set(query.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = sum(1 for word in chunk_words if word in query_words)
        scores.append(score)

    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    top_chunks = [chunks[i] for i in ranked_indices[:top_k]]
    return top_chunks


# Option 2: Semantic search (recommended for good accuracy)
# Initialize the model once when the module is loaded
model = SentenceTransformer('all-MiniLM-L6-v2')


def find_relevant_chunks_semantic(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Finds the most semantically similar chunks using sentence embeddings."""
    corpus_embeddings = model.encode(chunks, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Use PyTorch for the dot product calculation for semantic search
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

    top_chunks = [chunks[hit['corpus_id']] for hit in hits]
    return top_chunks


# --- Main Application Block (for testing) ---
if __name__ == "__main__":
    print("--- Testing Utility Functions ---")

    pdf_file_path = "case_studies.pdf"
    document_pages = extract_text_from_pdf(pdf_path=pdf_file_path)

    if document_pages:
        # The pages are our chunks now
        text_chunks = document_pages
        print(f"Document chunked into {len(text_chunks)} pieces.")

        # 2. Define a test query
        test_query = "What are the duties of a judge regarding financial matters?"
        print(f"\nTest Query: '{test_query}'")

        # 3. Test Keyword Search
        print("\n--- Testing Keyword Search ---")
        keyword_results = find_relevant_chunks_keyword(test_query, text_chunks)
        for i, chunk in enumerate(keyword_results):
            print(f"Result {i + 1}:\n\"{chunk[:150].strip()}...\"\n")

        # 4. Test Semantic Search
        print("\n--- Testing Semantic Search ---")
        semantic_results = find_relevant_chunks_semantic(test_query, text_chunks)
        for i, chunk in enumerate(semantic_results):
            print(f"Result {i + 1}:\n\"{chunk[:150].strip()}...\"\n")
