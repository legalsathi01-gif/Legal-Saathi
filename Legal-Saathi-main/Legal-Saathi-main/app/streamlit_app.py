# app/streamlit_app.py
import streamlit as st
from PIL import Image
import os
import sys
import io

# --- Path Setup ---
# This is a crucial step to ensure the app can find the 'backend' module.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Backend Module Imports ---
# Use Streamlit's caching to load heavy AI models only once.
@st.cache_resource
def load_modules():
    """Loads all necessary backend modules and their models."""
    from backend import summarizer, utils, ocr, stt, tts
    return summarizer, utils, ocr, stt, tts

with st.spinner("Warming up the AI engines... This may take a moment."):
    summarizer, utils, ocr, stt, tts = load_modules()

# --- Page Configuration ---
st.set_page_config(
    page_title="Legal Saathi - Your Legal Document Assistant",
    page_icon="⚖️",
    layout="wide"
)

# --- Session State Initialization ---
# 'text_chunks' will store the list of text pieces (pages, paragraphs, etc.)
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []

# --- Helper Functions ---
def play_audio(text):
    """Converts text to audio and displays it in Streamlit."""
    try:
        audio_bytes = tts.text_to_mp3_bytes(text)
        st.audio(audio_bytes, format='audio/mp3')
    except Exception as e:
        st.error(f"Could not generate audio. Error: {e}")

# --- Main UI ---
st.title("⚖️ Legal Saathi")
st.markdown("Your AI-powered assistant for understanding legal documents and audio.")

st.sidebar.header("Upload Your File")
st.sidebar.markdown("Upload a document (PDF, PNG, JPG) or an audio file (MP3, WAV, OGG).")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=['pdf', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'ogg']
)

if uploaded_file:
    file_type = uploaded_file.type
    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Reset state on new file upload
    st.session_state.text_chunks = []
    if 'summary' in st.session_state: del st.session_state.summary
    if 'risks' in st.session_state: del st.session_state.risks

    # Process file based on its type
    with st.spinner(f"Processing '{uploaded_file.name}'..."):
        text_content = ""
        if file_type in ['image/png', 'image/jpeg']:
            image = Image.open(uploaded_file)
            text_content = ocr.ocr_image_with_preprocessing(image)
            # For images/audio, we treat the whole text as one chunk
            if text_content:
                st.session_state.text_chunks = [text_content]

        elif file_type == 'application/pdf':
            pdf_bytes = uploaded_file.getvalue()
            # This now returns a list of page texts (our chunks)
            st.session_state.text_chunks = utils.extract_text_from_pdf(stream=pdf_bytes)

        elif file_type in ['audio/mpeg', 'audio/wav', 'audio/ogg']:
            temp_dir = "temp_audio"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            text_content = stt.transcribe_audio(temp_path)
            os.remove(temp_path)
            if text_content:
                st.session_state.text_chunks = [text_content]

# --- Main Content Area ---
if st.session_state.text_chunks:
    # For display, join all chunks into a single string
    full_text = "\n\n".join(st.session_state.text_chunks)

    st.header("Extracted Content")
    with st.expander("Click to view the full text from your file"):
        st.text_area("", full_text, height=300)

    st.markdown("---")
    st.header("Analysis Tools")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Document Summary")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing the document..."):
                # Pass the list of chunks directly to the summarizer
                summary = summarizer.summarize_document(st.session_state.text_chunks)
                st.session_state.summary = summary
        
        if 'summary' in st.session_state:
            st.success("Summary Generated!")
            st.write(st.session_state.summary)
            if st.button("Listen to Summary"):
                play_audio(st.session_state.summary)

    with col2:
        st.subheader("⚠️ Risk Identification")
        if st.button("Identify Risky Clauses"):
            with st.spinner("Analyzing for potential risks..."):
                # Join the first few chunks for risk analysis to provide enough context
                text_for_risk_analysis = "\n".join(st.session_state.text_chunks[:5])
                risks = summarizer.extract_risky_clauses(text_for_risk_analysis)
                st.session_state.risks = risks

        if 'risks' in st.session_state:
            st.warning("Potential Risks Identified!")
            st.write(st.session_state.risks)
            if st.button("Listen to Risk Analysis"):
                play_audio(st.session_state.risks)

    st.markdown("---")
    st.header("❓ Ask a Question")
    st.markdown("Ask a specific question about the document content.")

    query = st.text_input("Enter your question here:")

    if query:
        with st.spinner("Searching for the answer..."):
            # The pages/chunks from the PDF are perfect for semantic search
            if st.session_state.text_chunks:
                relevant_chunks = utils.find_relevant_chunks_semantic(
                    query, 
                    st.session_state.text_chunks, 
                    top_k=3
                )
                st.subheader("Relevant Information Found:")
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"> {chunk.strip()}")
                    st.markdown("---")
            else:
                st.warning("Could not find any text to search within.")
else:
    st.info("Please upload a file using the sidebar to get started.")


### How to Run the App

