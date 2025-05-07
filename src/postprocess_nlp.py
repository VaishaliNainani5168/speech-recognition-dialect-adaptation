from transformers import pipeline

def correct_transcription(transcription):
    # Use a more appropriate model for grammar correction
    try:
        nlp_pipeline = pipeline("text2text-generation", 
                               model="facebook/bart-large-cnn",
                               max_length=512)
        
        # Limit input length to avoid issues
        if transcription and len(transcription) > 1000:
            transcription = transcription[:1000]
            
        # Add prompt to guide correction
        prompt = f"Correct this text: {transcription}"
        
        corrected_text = nlp_pipeline(prompt, max_length=512)[0]['generated_text']
        
        # Remove the prompt if it appears in the output
        if corrected_text.startswith("Correct this text:"):
            corrected_text = corrected_text.replace("Correct this text:", "").strip()
            
        return corrected_text
    except Exception as e:
        print(f"Error during text correction: {e}")
        return transcription  # Return original if correction fails
