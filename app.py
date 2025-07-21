import os
import io
import time
import base64
import requests
import openai
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ------------------ CONFIGURATION ------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
OCR_URL = AZURE_ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"
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


# ------------------ AZURE OCR ------------------
def extract_handwritten_text(image_bytes):
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(OCR_URL, headers=headers, data=image_bytes)
    while response.status_code == 429:
        st.warning("Azure OCR rate limit hit. Waiting 10 seconds...")
        time.sleep(10)
        response = requests.post(OCR_URL, headers=headers, data=image_bytes)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]

    while True:
        final_response = requests.get(operation_url, headers=headers)
        result = final_response.json()
        if result.get("status") == "succeeded":
            break
        elif result.get("status") == "failed":
            raise Exception("OCR failed.")
        time.sleep(1)

    lines = []
    for read_result in result["analyzeResult"]["readResults"]:
        for line in read_result["lines"]:
            lines.append(line["text"])

    return "\n".join(lines)


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
