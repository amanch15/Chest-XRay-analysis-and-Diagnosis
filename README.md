# Medical RAG-Scanner

A Retrieval-Augmented Generation (RAG) system for chest X-ray analysis using NIH ChestX-ray14, CLIP vision encoders, FAISS vector search, and Llama 3.2 Vision.

---

## Project Structure

```
MAJOR PROJECT/
├── data/
│   ├── raw/            ← Original NIH X-ray images (.png)
│   ├── processed/      ← CLAHE-enhanced & resized (224x224)
│   └── metadata/       ← Data_Entry_2017.csv + processed_metadata.csv
├── src/
│   ├── utils.py        ← Logger & config loader
│   ├── data_loader.py  ← Week 1: CLAHE preprocessing pipeline
│   ├── vision_encoder.py ← Week 2: CLIP embedding extraction
│   ├── vector_db.py    ← Week 3: FAISS index build & search
│   └── generator.py    ← Week 4: Llama 3.2 RAG pipeline
├── notebooks/
│   ├── 01_eda.ipynb    ← Exploratory Data Analysis
│   └── 02_embeddings.ipynb ← Embedding visualization & testing
├── models/
│   ├── checkpoints/    ← Fine-tuned weights (if any)
│   ├── embeddings.npy  ← CLIP vectors for all processed images
│   └── faiss_index.bin ← Searchable vector index (Week 3)
├── app/
│   └── main.py         ← Streamlit frontend (Week 5)
├── config.yaml         ← All hyperparameters & paths
└── requirements.txt    ← Python dependencies
```

---

## Setup

### 1. Create & activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place your dataset
- Copy NIH X-ray images → `data/raw/`
- Copy `Data_Entry_2017.csv` → `data/metadata/`

---

## Week 1 — Data Preprocessing

```bash
python src/data_loader.py
```
- Applies CLAHE contrast enhancement
- Resizes all images to 224×224
- Outputs to `data/processed/`
- Saves `data/metadata/processed_metadata.csv`

---

## Week 2 — Vision Encoder

```bash
python src/vision_encoder.py
```
- Loads CLIP ViT-B/32 from HuggingFace
- Encodes all processed images → 512-dim vectors
- Saves `models/embeddings.npy` and `models/image_paths.txt`

---

## Configuration

Edit `config.yaml` to change:
- `preprocessing.max_samples` — limit images processed (set to `-1` for all)
- `encoder.model_name` — swap CLIP variant
- `faiss.k_neighbors` — number of similar cases to retrieve

---

## Tech Stack

| Component       | Technology                        |
|-----------------|-----------------------------------|
| Vision Encoder  | CLIP ViT-B/32 (HuggingFace)      |
| Vector DB       | FAISS                             |
| LLM             | Llama 3.2 11B Vision              |
| Preprocessing   | OpenCV CLAHE                      |
| Frontend        | Streamlit                         |
| Dataset         | NIH ChestX-ray14 (112k images)   |
