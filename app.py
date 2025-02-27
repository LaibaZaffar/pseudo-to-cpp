import streamlit as st
import torch
from transformers import T5ForConditionalGeneration
import sentencepiece as spm
import os
import zipfile
import gdown
from safetensors.torch import load_file  # Import Safetensors loader

MODEL_URL = "https://drive.google.com/uc?id=1nFx0Kqe30sxxmpGZNrJbB_QbSTMLLweX"

def download_and_extract():
    if not os.path.exists("models"):
        print("Downloading model...")
        gdown.download(MODEL_URL, "models.zip", quiet=False)

        print("Extracting model...")
        with zipfile.ZipFile("models.zip", "r") as zip_ref:
            zip_ref.extractall(".")

download_and_extract()

# def load_model(model_path="model"):
#     """Load the trained T5 model and tokenizer."""
#     model = T5ForConditionalGeneration.from_pretrained(model_path)
#     model.eval()
#     sp = spm.SentencePieceProcessor()
#     sp.load(f"{model_path}/spm_tokenizer.model")
#     return model, sp

def load_model(model_path="models"):
    """Load the trained T5 model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained(model_path, use_safetensors=True)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_path}/spm_tokenizer.model")
    
    return model, sp

def generate_code(model, tokenizer, text, max_length=256):
    """Generate C++ code from pseudocode."""
    input_ids = tokenizer.encode_as_ids(text)
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_length=max_length)
    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text

# Load model & tokenizer
model, tokenizer = load_model()

# Streamlit UI
st.title("Pseudocode to C++ Code Generator")
st.write("Enter your pseudocode and generate C++ code!")

# Input text
user_input = st.text_area("Enter pseudocode:", "")

if st.button("Generate Code"):
    if user_input.strip():
        generated_code = generate_code(model, tokenizer, user_input)
        st.subheader("Generated C++ Code:")
        st.code(generated_code, language="cpp")
    else:
        st.warning("Please enter valid pseudocode.")
