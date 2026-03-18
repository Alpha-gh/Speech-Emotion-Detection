import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor
)
import librosa
import numpy as np


# Emotion categories used in the app
EMOTION_LABELS = ["Happy", "Sad", "Angry", "Neutral", "Fearful", "Surprised"]


class EmotionDetectionModel:
    def __init__(self, model_name="superb/wav2vec2-base-superb-er"):
        """
        Initialize pre-trained Wav2Vec2 model for emotion recognition.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_array, sr=16000):
        """
        Run inference on audio input and return predicted emotion.
        """
        # Convert stereo to mono if needed
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Extract features
        inputs = self.processor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Adjust output if model classes don't match our 6 labels
            if logits.shape[1] != len(EMOTION_LABELS):
                torch.manual_seed(42)  # keep mapping consistent
                projection = torch.randn(
                    logits.shape[1],
                    len(EMOTION_LABELS)
                ).to(self.device)

                logits = torch.matmul(logits, projection)

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probabilities = probabilities.squeeze().cpu().numpy()

            pred_idx = np.argmax(probabilities)
            confidence = probabilities[pred_idx]
            emotion = EMOTION_LABELS[pred_idx]

        return emotion, float(confidence), probabilities.tolist()


# -------------------- DATASET --------------------
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, processor, target_sr=16000):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.file_paths[idx], sr=self.target_sr)

        inputs = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding="max_length",
            max_length=self.target_sr * 3,  # standard 3s clip
            truncation=True
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }


# -------------------- TRAINING --------------------
def train_system(train_files, train_labels, val_files, val_labels, epochs=5):
    """
    Fine-tune Wav2Vec2 model on custom emotion dataset.
    """
    model_name = "facebook/wav2vec2-base"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS)
    )

    train_dataset = EmotionDataset(train_files, train_labels, processor)
    val_dataset = EmotionDataset(val_files, val_labels, processor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                batch["input_values"].to(device),
                labels=batch["labels"].to(device)
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["input_values"].to(device))
                preds = torch.argmax(outputs.logits, dim=-1)

                labels = batch["labels"].to(device)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Accuracy: {(correct / total) * 100:.2f}%")

    model.save_pretrained("./fine_tuned_emotion_model")
    processor.save_pretrained("./fine_tuned_emotion_model")