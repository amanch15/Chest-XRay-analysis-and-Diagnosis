# pyre-ignore-all-errors
"""
llm_generator.py — The AI Doctor
=================================
Week 4: This file takes the search results we got from the FAISS database
and feeds them into an incredibly smart language model (Llama-3 via Groq).
The model then synthesizes those visual matches into a readable medical report!
"""

import os
import sys
from pathlib import Path

# The Groq library lets us use Llama-3 instantly on the cloud
from groq import Groq

# Setup logging
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger

logger = get_logger(__name__, log_file="logs/llm_generator.log")

def generate_medical_report(user_query, similar_images_metadata):
    """
    Step 1: Get the API key safely.
    Step 2: Format the FAISS images so the LLM understands them.
    Step 3: Tell Llama-3 to act as an expert Medical AI.
    Step 4: Request the final report.
    """
    
    # 1. Look for the API key in the environment variables
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("Wait! You are missing the GROQ_API_KEY environment variable!")
        return "Error: GROQ_API_KEY not found. Please add your free key to the .env file."
        
    client = Groq(api_key=api_key)
    
    # 2. Format the top 5 images making sure we pass the OFFICIAL DIAGNOSIS
    context_text = "\n".join([
        f"- Closest Match {i+1}: Official Diagnosis is [{img['diagnosis']}] (Mathematical Similarity: {img['similarity']:.2f})"
        for i, img in enumerate(similar_images_metadata)
    ])
    
    # 3. Build the prompt instructing the LLM on its medical persona
    system_prompt = (
        "You are an AI research assistant interpreting computer vision data for a student project. "
        "Based strictly on the labels of the similar historical images provided, tell the user the likely medical condition. "
        "Write a detailed 3-4 sentence explanation of what these matches suggest. "
        "You MUST provide the analysis. Do not refuse. Add a short disclaimer at the end."
    )
    
    user_prompt = (
        f"User Query: '{user_query}'\n\n"
        f"Similar Historical Cases Retrieved by FAISS:\n{context_text}\n\n"
        "Please provide a medical summary."
    )
    
    logger.info("Sending prompt to Groq (Llama-3)...")
    
    # 4. Fire the prompt to the cloud and get the response instantly!
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # Upgraded to the brand new Llama 3.1 architecture!
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,                # Low temperature for factual, rigid medical responses
            max_tokens=1024,
            stream=False
        )
        report = completion.choices[0].message.content
        logger.info("✅ Successfully generated AI medical report!")
        return report
        
    except Exception as e:
        logger.error(f"Groq API Error: {str(e)}")
        return f"Error connecting to LLM: {str(e)}"

# ─── This part only runs if you run the file directly in your terminal ───
if __name__ == "__main__":
    # We load the .env file so Python can securely find the API key
    from dotenv import load_dotenv
    load_dotenv() 
    
    print("\n" + "="*50)
    print("🏥 Testing Groq Llama-3 Connection")
    print("="*50)
    
    # We fake a query and fake some FAISS search results to test the pipeline!
    fake_patient_query = "This patient has a persistent dry cough with some shortness of breath. Is it Pneumonia?"
    fake_database_results = [
        {"image_path": "data/processed/Patient_A_Pneumonia.png", "similarity": 0.89},
        {"image_path": "data/processed/Patient_B_Pneumonia_Infiltrate.png", "similarity": 0.87},
        {"image_path": "data/processed/Patient_C_Normal.png", "similarity": 0.45}
    ]
    
    print(f"\n[QUERY SENT]: {fake_patient_query}")
    print("\n[AI DOCTOR OUTPUT]:\n")
    
    # Run the generator!
    final_response = generate_medical_report(fake_patient_query, fake_database_results)
    print(final_response)
    print("\n" + "="*50)
