import torch
import pretty_midi
import numpy as np
import os
import subprocess
import sys
from music_generation.model import NoteLSTM

# Yeni: Besteciye özel motifler
COMPOSER_MOTIFS = {
    "Beethoven": [60, 62, 64, 67, 69, 67, 65, 64],
    "Mozart": [60, 62, 64, 65, 67, 69, 71, 72],
    "Bach": [60, 64, 67, 71, 67, 64, 60],
    "Chopin": [60, 63, 67, 70, 67, 63, 60],
    "Debussy": [60, 62, 65, 69, 72, 74]
}

def generate_music_lstm(output_path="generated_from_model.mid", composer="Mozart"):
    # Besteciye uygun motif seç
    motif = COMPOSER_MOTIFS.get(composer, COMPOSER_MOTIFS["Mozart"])
    
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    melody = motif * 6  # Motifi genişlet

    start = 0
    duration = 0.4

    for pitch in melody:
        velocity = np.random.randint(80, 110)
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=start + duration)
        piano.notes.append(note)
        start += duration

    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"Saved MIDI to: {output_path}")
    return output_path

def convert_midi_to_mp3(midi_path, mp3_path="output.mp3"):
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, ".."))
        wav_path = os.path.join(project_root, "temp.wav")
        midi_abs_path = os.path.abspath(midi_path)

        if os.path.exists(wav_path):
            os.remove(wav_path)

        os.chdir(project_root)

        cmd = f"bin\\fluidsynth.exe -ni FluidR3_GM.sf2 {midi_abs_path} -F temp.wav -r 44100"
        exit_code = os.system(cmd)
        print(f"Fluidsynth exit code: {exit_code}")

        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
            return wav_path
        else:
            return synthesize_with_pretty_midi(midi_path, wav_path)

    except Exception as e:
        print(f"❌ Error in audio conversion: {e}")
        return synthesize_with_pretty_midi(midi_path, wav_path)

def synthesize_with_pretty_midi(midi_path, wav_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio = midi_data.synthesize(fs=44100)

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9

        import scipy.io.wavfile
        scipy.io.wavfile.write(wav_path, 44100, (audio * 32767).astype(np.int16))
        return wav_path
    except Exception as e:
        print(f"pretty_midi synthesis failed: {e}")
        return None
