import streamlit as st
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
import torch
import tempfile
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from src
from src.postprocess_nlp import correct_transcription

# Load the model and processor with caching
@st.cache_resource
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained("models/wav2vec2_dialect")
    model = AutoModelForCTC.from_pretrained("models/wav2vec2_dialect")
    return processor, model

# Transcribe audio
def transcribe_audio(audio_path, processor, model):
    waveform, sample_rate = torchaudio.load(audio_path)
    st.write(f"Original Sample Rate: {sample_rate}")

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    try:
        inputs = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# Streamlit UI
st.title("🎙️ Speech-to-Text with Wav2Vec2")
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

# Add dialect selection
st.sidebar.title("Settings")
dialect = st.sidebar.selectbox(
    "Select dialect (for fine-tuning)",
    ["General American", "Indian English", "British English", "Australian English"]
)

# Display selected dialect
st.write(f"Selected dialect: {dialect}")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_audio_path = tmp_file.name

    processor, model = load_model_and_processor()
    transcription = transcribe_audio(tmp_audio_path, processor, model)

    if transcription:
        st.success("Transcription:")
        st.write(transcription)
        
        with st.expander("Show corrected transcription"):
            try:
                corrected = correct_transcription(transcription)
                if corrected and len(corrected) > 5:  # Basic validation
                    st.write(corrected)
                else:
                    st.error("Correction failed to produce valid output")
                    st.write("Original transcription: " + transcription)
            except Exception as e:
                st.error(f"Error during correction: {str(e)}")
                st.write("Original transcription: " + transcription)

    os.remove(tmp_audio_path)
