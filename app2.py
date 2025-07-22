import os
import io
import base64
import openai
import streamlit as st
from dotenv import load_dotenv

# ------------------ CONFIGURATION ------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# ------------------ IMAGE TO BASE64 ------------------
def image_to_base64_url(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"


# ------------------ GPT-4o VISION TRANSCRIPTION ------------------
def transcribe_with_gpt4_vision(image_bytes):
    image_url = image_to_base64_url(image_bytes)

    prompt = """You are a transcription expert. Carefully review this handwritten image and transcribe the text exactly as it appears. Do not invent words. Keep original spelling and punctuation, even if incorrect. If you cannot read a word, write [illegible].

Output only the transcription with no explanation or additional commentary."""

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
st.title("Mission Journal Transcriber (GPT-4o Vision Only)")

uploaded_files = st.file_uploader("Upload your handwritten journal images (JPEG, PNG)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files and st.button("Transcribe Images"):
    progress_bar = st.progress(0)
    all_clean_text = ""

    for idx, uploaded_file in enumerate(uploaded_files):
        # Step 1: Read image bytes directly (no compression)
        image_bytes = uploaded_file.read()

        # Step 2: GPT-4o Direct Transcription
        st.write(f"Transcribing: {uploaded_file.name} with GPT-4o Vision...")
        transcription = transcribe_with_gpt4_vision(image_bytes)

        # Step 3: Combine outputs
        all_clean_text += f"# {uploaded_file.name}\n\n{transcription}\n\n---\n\n"
        progress_bar.progress((idx + 1) / len(uploaded_files))

    st.success("All files transcribed.")
    st.text_area("Final Combined Transcriptions", all_clean_text, height=300)
    st.download_button("Download as TXT", all_clean_text, file_name="combined_journal.txt")
