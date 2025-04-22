import streamlit as st
import torch
import soundfile as sf
import numpy as np
import io
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import tempfile
import os
import librosa  # Added for better audio handling

# Set page config
st.set_page_config(
    page_title="Speech Recognition System",
    page_icon="🎙️",
    layout="centered"
)

# Function to load model
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

# Enhanced audio processing function
def process_audio(audio_file, processor, model):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_filename = tmp_file.name

    try:
        # Use librosa for reliable audio loading and resampling
        speech_array, sampling_rate = librosa.load(tmp_filename, sr=16000, mono=True)
        
        # Check if audio is too short and pad if necessary
        min_samples = 480  # minimum required for Wav2Vec2 model
        if len(speech_array) < min_samples:
            padding = np.zeros(min_samples - len(speech_array))
            speech_array = np.concatenate([speech_array, padding])
        
        # Process the audio
        inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        # Get predicted ids and convert to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        return transcription

    except Exception as e:
        st.error(f"Detailed error: {str(e)}")
        return f"Error: {str(e)}"
        
    finally:
        # Clean up temp files
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)

def main():
    # Load the model
    with st.spinner("Loading Speech Recognition Model..."):
        processor, model = load_model()

    # App title and description
    st.title("🎙️ Speech Recognition System")
    st.markdown("""
    This application transcribes speech from audio files using the Wav2Vec2 model.
    Upload an audio file to get started!
    """)

    # Upload audio
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'm4a', 'flac'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')

        if st.button("Transcribe Uploaded Audio"):
            with st.spinner("Transcribing..."):
                try:
                    transcription = process_audio(uploaded_file, processor, model)

                    # Display results
                    with st.expander("Transcription Result", expanded=True):
                        st.markdown("### Transcription:")
                        st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'>{transcription}</div>", unsafe_allow_html=True)

                        # Add download button for transcript
                        transcript_bytes = transcription.encode()
                        st.download_button(
                            label="Download Transcript",
                            data=io.BytesIO(transcript_bytes),
                            file_name="transcript.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")

    # Model info
    with st.expander("About this Speech Recognition System"):
        st.markdown("""
        ### Technical Details
        - **Model**: Facebook's Wav2Vec2 (base-960h)
        - **Language**: English
        - **Architecture**: Convolutional feature encoder + Transformer
        - **Training Data**: 960 hours of LibriSpeech

        ### Limitations
        - Best performance on clear speech with minimal background noise
        - Optimized for English language
        - May struggle with heavily accented speech or uncommon words

        ### Project for AIML Internship at CODTECH
        """)

if __name__ == "__main__":
    main()
