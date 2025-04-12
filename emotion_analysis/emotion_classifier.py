import pretty_midi
import numpy as np

def extract_features(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)
    
    if not midi.instruments:
        raise ValueError("No instruments found in MIDI file.")
    
    instrument = midi.instruments[0]  # Assume single-instrument melody
    if not instrument.notes:
        raise ValueError("No notes found in the instrument.")

    pitches = [note.pitch for note in instrument.notes]
    velocities = [note.velocity for note in instrument.notes]
    durations = [note.end - note.start for note in instrument.notes]

    features = {
        "avg_pitch": np.mean(pitches),
        "pitch_var": np.var(pitches),
        "avg_velocity": np.mean(velocities),
        "avg_duration": np.mean(durations),
        "tempo": midi.estimate_tempo()
    }

    return features

def classify_emotion(features):
    if features["avg_pitch"] < 60 and features["avg_duration"] > 0.5:
        return "melancholic"
    elif features["pitch_var"] > 120:
        return "intense"
    elif features["avg_velocity"] > 110:
        return "triumphant"
    elif features["avg_duration"] > 0.8:
        return "peaceful"
    else:
        return "joyful"

if __name__ == "__main__":
    midi_path = "../music_generation/generated_from_model.mid"
    try:
        features = extract_features(midi_path)
        emotion = classify_emotion(features)
        print(f"Detected Emotion: {emotion}")
    except Exception as e:
        print(f"Error: {e}")
