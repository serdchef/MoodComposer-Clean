import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def assign_emotion(row):
    """Assign emotion based on musical features using a rule-based approach"""
    emotion = "unknown"
    
    # Mode (major/minor) is the primary indicator
    if pd.notna(row.get('mode')):
        is_major = row['mode'] == 'major'
        
        # Tempo is a secondary factor
        tempo = row.get('tempo')
        if pd.notna(tempo):
            if is_major and tempo > 120:
                emotion = "happy"
            elif is_major and tempo <= 120:
                emotion = "calm"
            elif not is_major and tempo > 120:
                emotion = "angry"
            elif not is_major and tempo <= 120:
                emotion = "sad"
    
    # Pitch mean and pitch range as additional factors
    pitch_mean = row.get('pitch_mean')
    pitch_range = row.get('pitch_range')
    
    if pd.notna(pitch_mean) and pd.notna(pitch_range):
        if pitch_mean > 70 and emotion in ["happy", "unknown"]:
            emotion = "bright"
        elif pitch_mean < 55 and emotion in ["sad", "unknown"]:
            emotion = "dark"
        
        # Wide pitch range often indicates a "dramatic" character
        if pitch_range > 40 and emotion not in ["calm"]:
            emotion = "dramatic"
    
    # Calculate intensity score (0-100)
    intensity = 50  # Default intensity
    
    note_density = row.get('note_density')
    if pd.notna(note_density):
        # Higher note density increases intensity
        intensity += min(25, note_density * 2)
    
    if pd.notna(pitch_range):
        # Wider pitch range increases intensity
        intensity += min(25, pitch_range / 2)
    
    # Ensure intensity is within [0, 100]
    intensity = max(0, min(100, intensity))
    
    return {
        "primary_emotion": emotion,
        "intensity": intensity
    }

def main():
    parser = argparse.ArgumentParser(description="Assign emotions to music based on features")
    parser.add_argument("--input", default="music_features.csv", 
                      help="Input CSV file with music features")
    parser.add_argument("--output", default="music_emotions.csv",
                      help="Output CSV file with emotion labels")
    parser.add_argument("--plot", action="store_true",
                      help="Generate emotion distribution plot")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        print("Please run file.py with --detailed flag first to generate the features file.")
        print("Example: python file.py --detailed --csv music_features.csv")
        return
    
    # Load music features from CSV
    print(f"Loading music features from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} music entries.")
    
    # Apply emotion assignment for each row
    print("Assigning emotions based on musical features...")
    emotions = df.apply(assign_emotion, axis=1)
    
    # Extract emotion and intensity values
    df['emotion'] = [e['primary_emotion'] for e in emotions]
    df['intensity'] = [e['intensity'] for e in emotions]
    
    # Save results to CSV
    print(f"Saving results to {args.output}...")
    df.to_csv(args.output, index=False)
    
    # Display basic statistics
    emotion_counts = df['emotion'].value_counts()
    print("\nEmotion distribution:")
    print(emotion_counts)
    
    # Calculate average intensity per emotion
    intensity_by_emotion = df.groupby('emotion')['intensity'].mean().round(1)
    print("\nAverage intensity by emotion:")
    print(intensity_by_emotion)
    
    # Optional: Create visualization
    if args.plot:
        plt.figure(figsize=(12, 10))
        
        # Plot emotion distribution
        plt.subplot(2, 1, 1)
        emotion_counts.plot(kind='bar')
        plt.title('Distribution of Emotions in Music Collection')
        plt.ylabel('Number of Pieces')
        plt.xlabel('Emotion')
        
        # Plot average intensity by emotion
        plt.subplot(2, 1, 2)
        intensity_by_emotion.plot(kind='bar', color='orange')
        plt.title('Average Intensity by Emotion')
        plt.ylabel('Intensity (0-100)')
        plt.xlabel('Emotion')
        
        plt.tight_layout()
        plt.savefig('emotion_analysis.png')
        print("\nEmotion analysis plot saved to 'emotion_analysis.png'")
        plt.show()

if __name__ == "__main__":
    main()
