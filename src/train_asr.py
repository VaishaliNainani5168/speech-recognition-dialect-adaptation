import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset

# Load dataset and model
def fine_tune_asr_model():
    dataset = load_dataset("common_voice", "en", split="train[:1%]")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # Preprocess the data
    def preprocess_data(batch):
        audio = batch["audio"]
        # Handle different input formats
        if isinstance(audio, dict) and "array" in audio:
            audio_array = audio["array"]
        else:
            audio_array = audio
        
        input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        batch["input_values"] = input_values
        
        # Convert text to label IDs
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        
        return batch

    dataset = dataset.map(preprocess_data, remove_columns=["audio"])

    # Training the model (for simplicity, using dummy parameters)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in dataset:
            optimizer.zero_grad()
            inputs = batch["input_values"]
            labels = batch["sentence"]

            # Forward pass
            outputs = model(input_values=inputs, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

    # Save the fine-tuned model
    model.save_pretrained("models/wav2vec2_dialect")
    processor.save_pretrained("models/wav2vec2_dialect")

if __name__ == "__main__":
    fine_tune_asr_model()
