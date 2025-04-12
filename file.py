import os
import mimetypes
import json
from collections import Counter
import music21
import numpy as np
import pandas as pd

def analyze_directory(path, detailed_midi=False):
    """Analyze the structure of a directory and its contents."""
    if not os.path.exists(path):
        return {"status": "error", "message": f"Path does not exist: {path}"}
    
    if os.path.isfile(path):
        return {"status": "single_file", "file_info": analyze_file(path, detailed_midi)}
    
    result = {
        "status": "directory",
        "directory": path,
        "file_count": 0,
        "midi_count": 0,
        "wav_count": 0,
        "csv_count": 0,
        "subdirectories": [],
        "file_types": Counter(),
        "total_size_bytes": 0,
        "midi_features": [],
        "sample_files": []
    }
    
    midi_files = []
    
    for root, dirs, files in os.walk(path):
        # Avoid duplicated subdirectories
        if result["subdirectories"].count(os.path.basename(root)) == 0 and root != path:
            result["subdirectories"].append(os.path.basename(root))
        
        result["file_count"] += len(files)
        
        for file in files:
            file_path = os.path.join(root, file)
            file_info = analyze_file(file_path, detailed=False)  # Basic info first
            
            # Update statistics
            result["file_types"][file_info["mime_type"]] += 1
            result["total_size_bytes"] += file_info["size_bytes"]
            
            # Count by file type
            if file_info["extension"].lower() == ".mid":
                result["midi_count"] += 1
                midi_files.append(file_path)
            elif file_info["extension"].lower() == ".wav":
                result["wav_count"] += 1
            elif file_info["extension"].lower() == ".csv":
                result["csv_count"] += 1
            
            # Add to samples (up to 10)
            if len(result["sample_files"]) < 10:
                result["sample_files"].append(file_info)
    
    # Analyze MIDI files if requested
    if detailed_midi and midi_files:
        print(f"Analyzing {min(20, len(midi_files))} MIDI files for musical features...")
        for midi_file in midi_files[:20]:  # Limit to first 20 files for speed
            try:
                midi_features = extract_midi_features(midi_file)
                if midi_features:
                    result["midi_features"].append(midi_features)
            except Exception as e:
                print(f"Error analyzing {midi_file}: {str(e)}")
    
    return result

def analyze_file(file_path, detailed=False):
    """Get information about a specific file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    extension = os.path.splitext(file_path)[1]
    
    file_info = {
        "name": os.path.basename(file_path),
        "path": file_path,
        "size_bytes": os.path.getsize(file_path),
        "mime_type": mime_type or "unknown/unknown",
        "extension": extension,
        "parent_dir": os.path.basename(os.path.dirname(file_path))
    }
    
    # If detailed analysis requested and it's a MIDI file
    if detailed and extension.lower() == ".mid":
        try:
            midi_features = extract_midi_features(file_path)
            if midi_features:
                file_info["musical_features"] = midi_features
        except Exception as e:
            file_info["analysis_error"] = str(e)
    
    return file_info

def extract_midi_features(midi_path):
    """Extract musical features from a MIDI file."""
    try:
        # Parse the MIDI file
        midi = music21.converter.parse(midi_path)
        
        # Extract basic features
        features = {
            "filename": os.path.basename(midi_path),
            "composer": os.path.basename(os.path.dirname(midi_path)),
            "duration_seconds": midi.duration.quarterLength * 60 / 100 if midi.duration else 0,
        }
        
        # Analyze key
        key_analysis = midi.analyze('key')
        if key_analysis:
            features["key"] = str(key_analysis)
            features["mode"] = key_analysis.mode  # 'major' or 'minor'
        
        # Get tempo
        tempo_marks = midi.metronomeMarkBoundaries()
        if tempo_marks:
            features["tempo"] = tempo_marks[0][2].number
        
        # Note statistics
        notes = midi.flat.notes
        if notes:
            pitches = [n.pitch.midi for n in notes if hasattr(n, 'pitch')]
            if pitches:
                features["pitch_mean"] = np.mean(pitches)
                features["pitch_std"] = np.std(pitches)
                features["pitch_range"] = max(pitches) - min(pitches)
            
            # Note density (notes per second)
            if features["duration_seconds"] > 0:
                features["note_density"] = len(notes) / features["duration_seconds"]
        
        # Instrument analysis
        instruments = []
        for part in midi.parts:
            instr = part.getInstrument()
            if instr:
                instruments.append(str(instr))
        
        features["instruments"] = instruments
        
        return features
    
    except Exception as e:
        print(f"Error processing {midi_path}: {str(e)}")
        return None

def save_midi_features_csv(midi_features, output_path="midi_features.csv"):
    """Save midi features to a CSV file."""
    if not midi_features:
        print("No MIDI features to save")
        return
    
    # return ifadesi burada olmamalı, silin
    
    # Convert list to DataFrame
    df = pd.DataFrame(midi_features)
    
    # Handle nested structures (like instruments list) - convert to string
    for col in df.columns:
        if df[col].apply(type).eq(list).any():
            df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"MIDI features saved to {output_path}")
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze music directory structure and MIDI features")
    parser.add_argument("--path", default=r"c:\Users\x\Desktop\musicarchive", 
                      help="Path to the music directory to analyze")
    parser.add_argument("--detailed", action="store_true",
                      help="Perform detailed musical analysis on MIDI files")
    parser.add_argument("--output", default="music_analysis.json",
                      help="Output file path for the analysis results")
    parser.add_argument("--csv", default="midi_features.csv",
                      help="Output CSV file for extracted MIDI features")
    
    args = parser.parse_args()
    
    print(f"Analyzing directory: {args.path}")
    result = analyze_directory(args.path, detailed_midi=args.detailed)
    
    # Save to JSON file
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Analysis completed and saved to {args.output}")
    
    # Save MIDI features to CSV if available
    if args.detailed and result["midi_features"]:
        save_midi_features_csv(result["midi_features"], args.csv)

