# backend/summarizer.py
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from Backend.utils import chunk_text, extract_text_from_pdf

# --- Model Initialization ---
# It's efficient to load models once when the module is imported.

# 1. Summarizer pipeline
# This model is great for creating concise summaries of documents.
# NOTE: Switched to a smaller model ('distilbart-cnn-6-6') to prevent download errors on unstable connections.
print("Loading summarization model...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
print("Summarization model loaded.")

# 2. Risk/Clause extraction model (Instruction-tuned T5)
# This model is designed to follow specific instructions, making it ideal for targeted extraction tasks.
print("Loading instruction-tuned model for risk extraction...")
risk_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
risk_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
print("Risk extraction model loaded.")


# --- Core Functions ---

def summarize_document(text: str) -> str:
    """
    Summarizes a long text by chunking it first and then summarizing each chunk.

    Args:
        text (str): The document text to be summarized.

    Returns:
        str: A consolidated string of all the chunk summaries.
    """
    print("Chunking text for summarization...")
    # Use the chunk_text utility to handle large documents
    chunks = chunk_text(text, chunk_size=2000, overlap=200)

    summaries = []
    print(f"Summarizing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        print(f"  - Summarizing chunk {i + 1}...")
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    return "\n\n---\n\n".join(summaries)


def extract_risky_clauses(text: str, max_length: int = 256) -> str:
    """
    Identifies and explains potentially risky clauses in a legal text using an instruction-tuned model.

    Args:
        text (str): The text to analyze.
        max_length (int): The maximum length of the generated output.

    Returns:
        str: The model's analysis of risky clauses.
    """
    prompt = (
        "You are a helpful legal assistant. Please read the following text carefully. "
        "Identify and list up to 5 clauses that could be considered risky for a regular person signing this document. "
        "For each clause, provide its title or a brief quote, and then explain in one simple sentence why it poses a risk.\n\n"
        "Text:\n"
        f"'{text}'"
    )

    print("Analyzing text for risky clauses...")
    inputs = risk_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    # Generate the output from the model
    output_tokens = risk_model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)

    # Decode and return the result
    return risk_tokenizer.decode(output_tokens[0], skip_special_tokens=True)


# --- Test Block ---
if __name__ == "__main__":
    print("\n--- Testing the Summarizer and Risk Extraction Module ---")

    # Use a real document for testing
    pdf_file_path = "case_studies.pdf"
    document_text = extract_text_from_pdf(pdf_file_path)

    if document_text:
        # --- Test 1: Document Summarization ---
        print("\n" + "=" * 50)
        print("PERFORMING DOCUMENT SUMMARIZATION")
        print("=" * 50)
        document_summary = summarize_document(document_text)
        print("\n--- CONSOLIDATED SUMMARY ---")
        print(document_summary)

        # --- Test 2: Risk Clause Extraction ---
        print("\n" + "=" * 50)
        print("PERFORMING RISK CLAUSE EXTRACTION")
        print("=" * 50)
        # We use only a slice of the text to keep the test quick and focused
        risky_clauses = extract_risky_clauses(document_text[:4000])  # Analyze first 4000 chars
        print("\n--- IDENTIFIED RISKY CLAUSES ---")
        print(risky_clauses)
    else:
        print(f"Could not read the document at '{pdf_file_path}'. Please ensure it exists.")

