import os
import io
import time
import requests
import openai
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
import os
from dotenv import load_dotenv
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


# Azure Configuration
OCR_URL = AZURE_ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"

def extract_handwritten_text(image_bytes):
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(OCR_URL, headers=headers, data=image_bytes)
    if response.status_code == 429:
        time.sleep(10)
        return extract_handwritten_text(image_bytes)
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


def clean_text_with_gpt4(raw_text):
    prompt = f"Clean up this text for spelling, punctuation, and formatting. Do not invent information:\n\n{raw_text}"
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# Streamlit UI
st.title("Mission Journal Transcriber (OCR + GPT-4)")

uploaded_files = st.file_uploader("Upload your handwritten journal images (JPEG, PNG)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files and st.button("Process Files"):
    all_clean_text = ""
    progress_bar = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files):
        image_bytes = uploaded_file.read()
        raw_text = extract_handwritten_text(image_bytes)
        cleaned_text = clean_text_with_gpt4(raw_text)
        all_clean_text += f"# {uploaded_file.name}\n\n{cleaned_text}\n\n---\n\n"
        progress_bar.progress((idx + 1) / len(uploaded_files))

    st.success("Done processing all files!")
    st.text_area("Combined Cleaned Text", all_clean_text, height=300)
    st.download_button("Download as TXT", all_clean_text, file_name="combined_journal.txt")
