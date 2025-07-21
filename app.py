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

    prompt = f"""You are a transcription expert. Below is the OCR result from the image. Your task is to carefully review the handwritten image and rewrite the transcription accurately. Correct any OCR mistakes. Only transcribe what is visibly written in the image. Do not invent words. Do not improve grammar or punctuation beyond what's written. OCR result for reference: /n/n{raw_text}"""

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
st.title("Mission Journal Transcriber (Azure OCR + Optional GPT-4o Vision Review)")

use_gpt4_vision = st.sidebar.checkbox("Use GPT-4o Vision to verify and correct Azure OCR?", value=True)

uploaded_files = st.file_uploader("Upload your handwritten journal images (JPEG, PNG)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files and st.button("Process Files"):
    progress_bar = st.progress(0)
    all_clean_text = ""

    for idx, uploaded_file in enumerate(uploaded_files):
        # Step 1: Compress
        image_bytes = compress_image_to_bytes(uploaded_file)

        # Step 2: Azure OCR
        raw_text = extract_handwritten_text(image_bytes)

        # Step 3: Optional GPT-4o Vision Check
        if use_gpt4_vision:
            st.write(f"Correcting: {uploaded_file.name} with GPT-4o Vision...")
            corrected_text = correct_text_with_gpt4_vision(image_bytes, raw_text)
        else:
            corrected_text = raw_text  # Trust Azure only

        # Step 4: Combine outputs
        all_clean_text += f"# {uploaded_file.name}\n\n{corrected_text}\n\n---\n\n"
        progress_bar.progress((idx + 1) / len(uploaded_files))

    st.success("All files processed.")
    st.text_area("Final Cleaned Combined Text", all_clean_text, height=300)
    st.download_button("Download as TXT", all_clean_text, file_name="combined_journal.txt")
