from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import re
from deepmultilingualpunctuation import PunctuationModel

# Load punctuation model globally
punctuation_model = PunctuationModel()

# Optional: define a cleaning function
def clean_transcription(text):
    # Convert to lowercase
    text = text.lower()

    # Fix common misspellings
    corrections = {
        "veshali": "vaishali",
        "udeippur": "udaipur",
        "rajastan": "rajasthan",
        "ta city": "a city"
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def transcribe_audio(audio_path):
    try:
        # Load the fine-tuned model and processor
        model = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2_dialect")
        processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2_dialect")
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        input_values = processor(audio, return_tensors="pt").input_values

        # Perform transcription
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Convert ids to text
        transcription = processor.decode(predicted_ids[0])
        
        # Clean the transcription
        cleaned_transcription = clean_transcription(transcription)
        
        # Add punctuation
        punctuated = punctuation_model.restore_punctuation(cleaned_transcription)

        return punctuated
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
