import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment

# Load model
model_name = "superb/wav2vec2-base-superb-er"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

labels = ["angry", "happy", "neutral", "sad"]

# 🔥 Universal audio loader
def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)  # supports ANY format
    audio = audio.set_channels(1)             # mono
    audio = audio.set_frame_rate(16000)       # 16kHz
    
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Normalize
    samples = samples / np.max(np.abs(samples))
    
    return samples

def predict_emotion(file_path):
    audio = load_audio(file_path)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1)[0][predicted_id].item()

    return labels[predicted_id], confidence


# 🔥 Test with ANY file
emotion, confidence = predict_emotion("test_audio.mp3")

print("Emotion:", emotion)
print("Confidence:", round(confidence, 2))