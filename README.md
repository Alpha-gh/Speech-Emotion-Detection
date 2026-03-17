# Speech Emotion Detection System

This project provides a complete speech emotion detection workflow:

- audio preprocessing and handcrafted feature extraction with `librosa`
- supervised model training for `happy`, `sad`, `angry`, `neutral`, `fearful`, and `surprised`
- confidence-scored emotion prediction for recorded or uploaded speech
- a simple Streamlit dashboard with waveform and spectrogram visualizations

## Project Structure

```text
.
|-- app.py
|-- train.py
|-- requirements.txt
|-- models/
|-- src/speech_emotion/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Format

You can train the model in either of these formats:

1. Folder structure

```text
data/
|-- angry/
|   |-- sample_01.wav
|-- fearful/
|-- happy/
|-- neutral/
|-- sad/
|-- surprised/
```

2. CSV manifest

```csv
path,emotion
clips/a1.wav,angry
clips/h1.wav,happy
clips/n1.wav,neutral
```

## Prepare Raw Datasets

If you download the raw `RAVDESS` or `CREMA-D` datasets, you can normalize them into the exact folder structure above with:

```bash
python prepare_dataset.py --source ravdess --input-dir downloads/RAVDESS --output-dir data
```

Or:

```bash
python prepare_dataset.py --source crema-d --input-dir downloads/CREMA-D --output-dir data
```

What the script does:

- copies supported files into `data/angry`, `data/fearful`, `data/happy`, `data/neutral`, `data/sad`, and `data/surprised`
- skips unsupported labels such as `calm`, `disgust`, and other classes outside the current app
- writes `data/manifest.csv` and `data/summary.txt`

Notes:

- `RAVDESS` supports all six emotions used by this app.
- `CREMA-D` does not include `surprised`, so that folder will remain empty unless you merge in another dataset.
- add `--mode move` if you want to move files instead of copying them.
- add `--include-unsupported-log` to also write `data/skipped_files.csv`.

## Train

```bash
python train.py --dataset-dir data
```

Or:

```bash
python train.py --manifest data/manifest.csv
```

Artifacts are saved to:

- `models/emotion_model.joblib`
- `models/label_encoder.joblib`
- `models/metrics.json`

## Run The Dashboard

```bash
streamlit run app.py
```

The app supports:

- file upload
- live microphone recording through Streamlit
- emotion prediction with confidence scores
- waveform and mel spectrogram views

## Notes

- Accuracy depends heavily on the dataset quality and speaker diversity.
- The included pipeline uses engineered acoustic features with a `RandomForestClassifier`, which is a practical baseline for hackathons and demos.
- If you want to push accuracy further, the same app can be extended to use CNN or LSTM models trained on spectrogram inputs.
