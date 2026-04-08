# pyre-ignore-all-errors
"""
vector_db.py — The Vector Database
===================================
Week 3: This file takes our giant list of AI numbers (embeddings)
and puts them into a special database called FAISS. FAISS allows us 
to search through 10,000 images in less than a millisecond!
"""

import os
import sys
import faiss
import numpy as np
from pathlib import Path

# Setup logging so we can see what's happening
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config

logger = get_logger(__name__, log_file="logs/vector_db.log")


def build_and_save_database(embeddings_file, save_location):
    """
    Step 1: Open the 20MB file containing our AI numbers.
    Step 2: Put them into a FAISS Database.
    Step 3: Save the database to the computer so we don't have to rebuild it.
    """
    if not os.path.exists(embeddings_file):
        logger.error("Could not find the embeddings! Did you run Week 2?")
        return
        
    logger.info("1. Loading the AI image numbers (embeddings) from your hard drive...")
    # Load the big grid of numbers that represent our images
    image_vectors = np.load(embeddings_file).astype(np.float32)
    
    total_images = len(image_vectors)
    vector_size = len(image_vectors[0])  # Should be 512 for CLIP model
    logger.info(f"   -> Successfully loaded {total_images} images, each compressed into {vector_size} numbers.")

    logger.info("2. Building the blazing fast FAISS Search Database...")
    # FAISS IndexFlatIP means it searches by "Cosine Similarity" (finding similar angles)
    search_database = faiss.IndexFlatIP(vector_size)
    
    # Push all 10,000 images into the database
    search_database.add(image_vectors)

    logger.info("3. Saving the finished database so we can load it instantly in the future...")
    os.makedirs(os.path.dirname(save_location), exist_ok=True)
    faiss.write_index(search_database, save_location)
    
    logger.info(f"✅ DONE! Database successfully saved to: {save_location}")


def search_for_similar_images(query_vector, database, image_paths, top_k=5):
    """
    This function will be used in Week 5 (The Website UI).
    When a doctor types a symptom, or uploads an X-Ray, we turn it into
    a 'query_vector' and ask the database for the TOP 5 most similar images.
    """
    # Make sure the query is the exact mathematical format the database expects
    query_vector = np.array(query_vector).astype(np.float32)
    
    # If it's just a 1D list of numbers, make it 2D so the database can read it
    if len(query_vector.shape) == 1:
        query_vector = np.expand_dims(query_vector, axis=0)

    # Ask the database to fetch the Top 5 closest matches instantly!
    distances, indices = database.search(query_vector, top_k)
    
    # Format the results so they are easy to read in our web app
    results = []
    
    # 'distances' = how similar they are (higher is better)
    # 'indices' = the row number of the winning image
    for similarity_score, row_number in zip(distances[0], indices[0]):
        results.append({
            "image_path": image_paths[row_number],
            "similarity": float(similarity_score)
        })
    
    return results


# ─── This part only runs if you run the file directly in your terminal ───
if __name__ == "__main__":
    # Load our settings file to figure out where the files are stored
    config_path = str(Path(__file__).resolve().parent.parent / "config.yaml")
    settings = load_config(config_path)
    
    # Grab the paths from the settings file
    emb_file = settings["paths"]["embeddings"]
    save_file = settings["paths"].get("faiss_index", "models/faiss_index.bin")
    
    
    logger.info("🚀 Starting the Database Builder...")
    build_and_save_database(emb_file, save_file)
