# pyre-ignore-all-errors
import streamlit as st
import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.utils import load_config
import faiss
import open_clip

from src.vision_encoder import (
    load_biomed_clip,
    load_densenet121,
    get_densenet_transform,
    encode_combined_image,
)
from src.vector_db import search_for_similar_images
from src.llm_generator import generate_medical_report
from src.reranker import load_text_tokenizer, cross_encoder_rerank
from src.explainability import generate_xai_heatmaps

# ─── Config ───────────────────────────────────────────────────────────────────
config = load_config(str(project_root / "config.yaml"))

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Medical X-RAYScanner", page_icon="🏥", layout="wide")
st.title("Medical RAG-Scanner: AI Radiologist")
st.markdown(
    "Upload a raw X-Ray. The **Dual Encoder** (DenseNet121 + BiomedCLIP) will find "
    "the most similar historical cases. The **Cross-Encoder Reranker** refines the "
    "results, and **Llama-3.3** writes a clinical diagnostic report."
)

# ─── Cached Model Loaders ─────────────────────────────────────────────────────

@st.cache_resource
def init_vision_models():
    """Load BiomedCLIP + DenseNet121 — both stay in RAM via cache."""
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["encoder"]["model_name"]

    biomed_model, biomed_preprocess = load_biomed_clip(model_name, device)
    densenet_model                  = load_densenet121(device)
    densenet_transform              = get_densenet_transform()

    return biomed_model, biomed_preprocess, densenet_model, densenet_transform, device


@st.cache_resource
def init_reranker_tokenizer():
    """Load BiomedCLIP tokenizer for the cross-encoder reranker."""
    model_name = config["encoder"]["model_name"]
    return load_text_tokenizer(model_name)


@st.cache_resource
def init_faiss_db():
    """Load the FAISS index and JSON metadata from disk."""
    index_path       = config["paths"].get("faiss_index", "models/faiss_index.bin")
    image_paths_file = config["paths"]["image_paths"].replace(".txt", ".json")

    index = faiss.read_index(index_path)

    import json
    with open(project_root / image_paths_file, "r") as f:
        image_paths = json.load(f)

    return index, image_paths


# ─── Warm Up ─────────────────────────────────────────────────────────────────
with st.spinner("Loading AI models — this only happens once..."):
    biomed_model, biomed_preprocess, densenet_model, densenet_transform, device = init_vision_models()
    tokenizer                   = init_reranker_tokenizer()
    faiss_index, image_db_paths = init_faiss_db()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Patient Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.sidebar.image(uploaded_file, caption="Uploaded Patient X-Ray")

    temp_path = str(project_root / "temp_upload.png")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.sidebar.button("Run Analysis", type="primary"):
        st.divider()

        # ── 1. Dual Encoder ───────────────────────────────────────────────────
        with st.spinner("Encoding X-Ray with Dual Encoder (DenseNet121 + BiomedCLIP)..."):
            query_vector = encode_combined_image(
                temp_path,
                biomed_model, biomed_preprocess,
                densenet_model, densenet_transform,
                device
            )

        # ── 2. XAI — Generate Heatmaps ────────────────────────────────────────
        with st.spinner("Generating XAI Explainability Maps (GradCAM + Saliency)..."):
            xai_results = generate_xai_heatmaps(
                temp_path,
                biomed_model, biomed_preprocess,
                densenet_model, densenet_transform,
                device
            )
            activated_region = xai_results["activated_region"]

        # Display XAI section
        st.subheader("🔍 XAI — Explainability Activation Maps")
        xai_cols = st.columns(3)
        with xai_cols[0]:
            original_pil = Image.open(temp_path).resize((224, 224))
            st.image(original_pil, caption="Original X-Ray", use_container_width=True)
        with xai_cols[1]:
            st.image(xai_results["densenet_overlay"],
                     caption="CNN Feature Map (DenseNet121)", use_container_width=True)
        with xai_cols[2]:
            st.image(xai_results["biomed_overlay"],
                     caption="ViT Saliency Map (BiomedCLIP)", use_container_width=True)

        st.info(f"🎯 **Primary AI Activation Region: {activated_region}** — "
                f"Both encoders show maximum focus on this anatomical zone.")
        st.divider()

        # ── 3. FAISS Search (initial_pull = 50) ───────────────────────────────
        initial_pull = config["database"].get("initial_pull", 50)
        final_top_k  = config["database"].get("final_rerank", 3)

        with st.spinner(f"Searching database for top {initial_pull} candidates..."):
            candidates = search_for_similar_images(
                query_vector, faiss_index, image_db_paths, top_k=initial_pull
            )

        # ── 3. Cross-Encoder Reranker ─────────────────────────────────────────
        alpha = config["reranker"].get("alpha", 0.6)
        with st.spinner("Cross-Encoder Reranking — aligning visual & clinical text evidence..."):
            results = cross_encoder_rerank(
                query_vector, candidates,
                biomed_model, tokenizer, device,
                alpha=alpha, top_k=final_top_k
            )

        # ── 4. Display Matched Images ─────────────────────────────────────────
        st.subheader("Closest Historical Matches (Cross-Encoder Reranked)")
        cols = st.columns(final_top_k)
        for i, match in enumerate(results):
            absolute_img_path = str(project_root / match["image_path"])
            with cols[i]:
                try:
                    matched_img = Image.open(absolute_img_path)
                    st.image(
                        matched_img,
                        caption=f"Historical Match #{i+1} — {match['diagnosis']}",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {repr(e)}. Path: {absolute_img_path}")

        # ── 5. LLM Report ─────────────────────────────────────────────────────
        st.subheader("AI Diagnostic Report")
        with st.spinner("Sending matches to Groq for medical synthesis..."):
            final_report = generate_medical_report(
                "Patient uploaded a new Chest X-Ray. What is the diagnosis?",
                results,
                xai_region=activated_region
            )
            st.success(final_report)

        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

else:
    st.info("👈 Please upload a chest X-Ray image in the left sidebar.")
