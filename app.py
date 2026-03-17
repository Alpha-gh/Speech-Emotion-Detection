import io
import time
import streamlit as st
from model import EmotionDetectionModel, EMOTION_LABELS
from utils import load_audio, plot_probability_distribution, get_emotion_suggestion, apply_custom_css

# Page Configuration
st.set_page_config(
    page_title="AI Speech Emotion Analyzer",
    page_icon="🎙️",
    layout="centered"
)

# Apply minimal, calm UI styling
apply_custom_css()

@st.cache_resource
def load_system_model():
    """Loads and caches the AI model to prevent reloading over multiple interactions."""
    return EmotionDetectionModel()

st.title("🎙️ AI Speech Emotion Analyzer")
st.markdown("""
<p style='color: #5b6a7a; font-size: 1.1rem; margin-bottom: 2rem;'>
Upload an audio file or record someone speaking. Our cutting-edge Wav2Vec2 transformer will instantly interpret the emotional tone of the voice.
</p>
""", unsafe_allow_html=True)

# Container for layout smoothness
with st.container():
    try:
        with st.spinner("🤖 Loading Neural Engine (this takes a few seconds)..."):
            model = load_system_model()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()

    st.markdown("### 📥 Input Source")
    tabs = st.tabs(["📁 Upload File", "🎤 Record Audio"])
    
    audio_bytes = None

    with tabs[0]:
        uploaded_file = st.file_uploader("", type=["wav", "mp3", "ogg"], 
                                         help="We support WAV, MP3, and OGG extensions.")
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()

    with tabs[1]:
        st.info("Record a short voice note. Make sure to allow microphone access.")
        audio_value = st.audio_input("Record directly from browser:")
        if audio_value is not None:
            audio_bytes = audio_value.read()

# Processing Engine
if audio_bytes is not None:
    st.markdown("---")
    st.audio(audio_bytes, format='audio/wav')
    
    # Process button
    if st.button("✨ Predict Emotion", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Futuristic loading animation loop
        for i in range(100):
            # Roughly 3.5 seconds total
            remaining_time = round((100 - i) * 0.035, 1)
            status_text.markdown(f"<p style='color: #00d4ff; font-weight: 600;'>Neural Analysis in Progress... Wait for ~{remaining_time} sec for the result.</p>", unsafe_allow_html=True)
            progress_bar.progress(i + 1)
            time.sleep(0.035)
            
        status_text.empty()
        progress_bar.empty()
        
        try:
            # Execution
            audio_file = io.BytesIO(audio_bytes)
            audio_array, sr = load_audio(audio_file)
            emotion, confidence, probabilities = model.predict(audio_array, sr)
            
            # Display 
            st.markdown("### 📊 Analysis Overview")
            col_res1, col_res2 = st.columns([1, 1.2])
            
            with col_res1:
                # Metric Card
                st.metric(
                    label="Dominant Emotion", 
                    value=emotion, 
                    delta=f"{confidence*100:.2f}% Confidence",
                    delta_color="normal"
                )
                
                # Intelligent Suggestion
                st.markdown("#### 💡 AI Companion Note")
                st.success(get_emotion_suggestion(emotion))
            
            with col_res2:
                # Distribution Chart
                st.markdown("<p style='text-align:center; font-weight: 600;'>Probability Matrix</p>", unsafe_allow_html=True)
                fig = plot_probability_distribution(probabilities, EMOTION_LABELS)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
        except Exception as e:
            import traceback
            st.error(f"⚠️ Failed to process audio. Please ensure format integrity or try another sample.\n\nError details:\n{e}")
            st.code(traceback.format_exc())

# Add footer
st.markdown("<div class='footer'>Made by <span>Team Visionaries</span></div>", unsafe_allow_html=True)
