# pyre-ignore-all-errors
import streamlit as st
import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add project root to python path so we can import our modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.utils import load_config
import faiss
from src.vision_encoder import load_clip_model, encode_image
from src.vector_db import search_for_similar_images
from src.llm_generator import generate_medical_report

# Load global configurations
config = load_config(str(project_root / "config.yaml"))

# ─── Page Setup ───
st.set_page_config(page_title="Medical RAG-Scanner", page_icon="🏥", layout="wide")
st.title("Medical RAG-Scanner: AI Radiologist")
st.markdown("Upload a raw X-Ray. The Vision AI will find the most mathematically similar historical cases from the database, and **Llama-3** will write a diagnostic report based on those matches.")

# ─── Safe Loading for Huge AI Models ───
# We use @st.cache_resource so the huge models stay perfectly running in RAM
# instead of reloading taking 5 seconds every single time you click a button!
@st.cache_resource
def init_vision_model():
    model_name = config["encoder"]["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_clip_model(model_name, device)
    return model, processor, device

@st.cache_resource
def init_faiss_db():
    index_path = config["paths"].get("faiss_index", "models/faiss_index.bin")
    image_paths_file = config["paths"]["image_paths"]
    
    # Pull the binary memory database
    index = faiss.read_index(index_path)
    
    # Pull the text list of image filenames
    with open(project_root / image_paths_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
        
    return index, image_paths

with st.spinner("Warming up AI Models..."):
    model, processor, device = init_vision_model()
    faiss_index, image_db_paths = init_faiss_db()


# ─── Sidebar Dashboard ───
st.sidebar.header("Patient Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Safely show the uploaded image
    st.sidebar.image(uploaded_file, caption="Uploaded Patient X-Ray")
    
    # Save the uploaded file temporarily so the CV2/Vision model can physically read it
    temp_path = str(project_root / "temp_upload.png")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.sidebar.button("Run AI Analysis", type="primary"):
        st.divider()
        
        # --- 1. Vision AI: Extract the Math ---
        with st.spinner("Analyzing image pixels with Vision Transformer..."):
            query_vector = encode_image(temp_path, model, processor, device)
            
        # --- 2. FAISS: Search Database ---
        with st.spinner("Searching 10,000 historical cases instantly in FAISS db..."):
            # We want the Top 3 closest matches
            results = search_for_similar_images(query_vector, faiss_index, image_db_paths, top_k=3)
            
        st.subheader("🔍 Closest Historical Matches Found")
        cols = st.columns(3)
        for i, match in enumerate(results):
            # Resolve absolute path to the processed image so Streamlit can draw it
            absolute_img_path = str(project_root / match["image_path"])
            
            with cols[i]:
                try:
                    matched_img = Image.open(absolute_img_path)
                    st.image(matched_img, caption=f"Match {i+1} | Similarity: {match['similarity_score']:.2f}", use_column_width=True)
                except Exception as e:
                    # Fallback if the path logic misaligned
                    st.error(f"Image Missing! Tried looking at: {absolute_img_path}")
        
        # --- 3. LLM Generator: The AI Doctor ---
        st.subheader("🤖 AI Diagnostic Report (Llama-3)")
        with st.spinner("Sending matches to Groq Supercomputer for medical synthesis..."):
            final_report = generate_medical_report("Patient uploaded a new Chest X-Ray. What is the diagnosis?", results)
            st.success(final_report)
            
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
else:
    st.info("👈 Please start by uploading a chest X-Ray image in the left sidebar.")
