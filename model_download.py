from transformers import AutoProcessor, AutoModelForCTC
import os

# Create the models directory if it doesn't exist
os.makedirs("models/wav2vec2_dialect", exist_ok=True)

# Download the model and processor from Hugging Face
model_name = "facebook/wav2vec2-base-960h" # or another suitable model
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCTC.from_pretrained(model_name)

# Save both the model and processor locally
model.save_pretrained("models/wav2vec2_dialect")
processor.save_pretrained("models/wav2vec2_dialect")

print("Model and processor successfully downloaded and saved to models/wav2vec2_dialect")
