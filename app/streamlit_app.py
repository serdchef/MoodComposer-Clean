import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from music_generation.generate_music import generate_music_lstm, convert_midi_to_mp3
from emotion_analysis.emotion_classifier import extract_features, classify_emotion
from image_generation.generate_cover import generate_cover_from_emotion

st.set_page_config(page_title="MoodComposer", layout="centered")
st.title("🎼 MoodComposer")
st.markdown("AI-generated classical music with emotion-aware album art 🎨")

# Composer style selection
composer = st.selectbox("🎹 Choose a classical composer style:", [
    "Mozart", "Beethoven", "Bach", "Chopin", "Debussy"
])

if st.button("🎶 Generate Music"):
    try:
        with st.spinner("🎵 Generating music..."):
            midi_path = generate_music_lstm(composer=composer)
            st.success("✅ Music generated")

        with st.spinner("🔊 Converting to audio..."):
            audio_path = convert_midi_to_mp3(midi_path)
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path)
                st.success("🔊 Audio ready")
                file_size = os.path.getsize(audio_path) / 1024
                st.info(f"📊 File Info: WAV format, {file_size:.1f} KB")
            else:
                st.warning("⚠️ Audio conversion failed")

        with st.spinner("🎭 Analyzing emotion..."):
            features = extract_features(midi_path)
            emotion = classify_emotion(features)
            st.info(f"🎭 Detected Emotion: **{emotion}**")

        with st.spinner("🎨 Generating cover art..."):
            image_path = generate_cover_from_emotion(emotion)
            st.image(image_path, caption=f"{emotion.capitalize()} Album Cover", use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error occurred: {str(e)}")
        st.code(f"{type(e).__name__}: {str(e)}")
