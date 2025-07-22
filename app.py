import os
import io
import time
import base64
import openai
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from google.cloud import vision
from google.oauth2 import service_account

# ------------------ CONFIGURATION ------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Google Cloud Vision - load credentials from Streamlit Secrets
credentials_info = st.secrets["gcp_service_account"]
google_credentials = service_account.Credentials.from_service_account_info(credentials_info)
vision_client = vision.ImageAnnotatorClient(credentials=google_credentials)
# ----------------------------------------------------


# ------------------ IMAGE COMPRESSION ------------------
def compress_image_to_bytes(uploaded_file, max_size_kb=4000):
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    output = io.BytesIO()
    quality = 85
    img.save(output, format="JPEG", quality=quality)
    while output.tell() > max_size_kb * 1024 and quality > 10:
        output = io.BytesIO()
        quality -= 5
        img.save(output, format="JPEG", quality=quality)
    output.seek(0)
    return output.read()


# ------------------ GOOGLE CLOUD VISION OCR ------------------
def extract_handwritten_text_google(image_bytes):
    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    text = response.full_text_annotation.text
    return text


# ------------------ IMAGE TO BASE64 ------------------
def image_to_base64_url(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"


# ------------------ GPT-4o VISION CLEANUP ------------------
def correct_text_with_gpt4_vision(image_bytes, raw_text):
    image_url = image_to_base64_url(image_bytes)

    prompt = f"""You are a transcription expert. Below is the OCR result from the image. 
Your task is to carefully review the handwritten image and rewrite the transcription accurately.
Correct any OCR mistakes. Only transcribe what is visibly written in the image.
Do not invent words. Do not improve grammar or punctuation beyond what's written.

OCR result for reference:
{raw_text}
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


# ------------------ STREAMLIT APP ------------------
st.title("Mission Journal Transcriber (Google OCR + Optional GPT-4o Vision Review)")

use_gpt4_vision = st.sidebar.checkbox("Use GPT-4o Vision to verify and correct OCR?", value=True)

uploaded_files = st.file_uploader("Upload your handwritten journal images (JPEG, PNG)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files and st.button("Process Files"):
    progress_bar = st.progress(0)
    all_clean_text = ""

    for idx, uploaded_file in enumerate(uploaded_files):
        # Step 1: Compress
        image_bytes = compress_image_to_bytes(uploaded_file)

        # Step 2: Google OCR
        raw_text = extract_handwritten_text_google(image_bytes)

        # Step 3: Optional GPT-4o Vision Check
        if use_gpt4_vision:
            st.write(f"Correcting: {uploaded_file.name} with GPT-4o Vision...")
            corrected_text = correct_text_with_gpt4_vision(image_bytes, raw_text)
        else:
            corrected_text = raw_text  # Trust Google OCR only

        # Step 4: Combine outputs
        all_clean_text += f"# {uploaded_file.name}\n\n{corrected_text}\n\n---\n\n"
        progress_bar.progress((idx + 1) / len(uploaded_files))

    st.success("All files processed.")
    st.text_area("Final Cleaned Combined Text", all_clean_text, height=300)
    st.download_button("Download as TXT", all_clean_text, file_name="combined_journal.txt")
