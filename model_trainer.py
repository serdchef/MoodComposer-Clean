import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def prepare_features(df):
    """Prepare feature matrix X and target vector y from dataset"""
    
    # Numerical features for prediction
    numeric_features = ['tempo', 'pitch_mean', 'pitch_std', 'pitch_range', 
                        'note_density', 'duration_seconds']
    
    # Categorical features for one-hot encoding
    categorical_features = ['mode', 'composer']
    
    # Remove unavailable features
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    # Create feature matrix X
    X = df[numeric_features].copy()
    
    # Fill missing values with column mean
    X = X.fillna(X.mean())
    
    # Add one-hot encoded categorical features
    if categorical_features:
        dummies = pd.get_dummies(df[categorical_features], drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    
    # Target variable y
    y = df['emotion']
    
    return X, y

def train_emotion_model(X, y):
    """Train a Random Forest Classifier to predict musical emotions"""
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': model.score(X_test_scaled, y_test),
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }

def plot_confusion_matrix(y_test, y_pred, output_file='confusion_matrix.png'):
    """Plot and save the confusion matrix as an image"""
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument("--input", default="music_emotions.csv", 
                      help="Input CSV file with music features and emotions")
    parser.add_argument("--model", default="emotion_model.joblib",
                      help="Output file for the trained model")
    parser.add_argument("--plot", action="store_true",
                      help="Generate confusion matrix plot")
    
    args = parser.parse_args()
    
    # Check input file existence
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        print("Please run emotion_labeler.py first to generate emotions file.")
        print("Example: python emotion_labeler.py")
        return
    
    # Load dataset
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} music entries with emotion labels.")
    
    # Prepare features and target
    print("Preparing features and target variables...")
    X, y = prepare_features(df)
    print(f"Features: {X.shape[1]} columns, {X.shape[0]} samples")
    
    # Warn if dataset is too small
    if len(df) < 10:
        print("Warning: Very small dataset. Consider collecting more data.")
    
    # Train model
    print("Training emotion classification model...")
    result = train_emotion_model(X, y)
    
    # Save model and scaler
    joblib.dump((result['model'], result['scaler']), args.model)
    print(f"Model saved to {args.model}")
    
    # Display results
    print(f"\nModel Accuracy: {result['accuracy']:.2f}")
    print("\nClassification Report:")
    print(classification_report(result['y_test'], result['y_pred']))
    
    # Feature importance
    print("\nFeature Importance:")
    importance = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feature, score in importance:
        print(f"{feature}: {score:.4f}")
    
    # Generate confusion matrix plot if requested
    if args.plot:
        output_file = plot_confusion_matrix(result['y_test'], result['y_pred'])
        print(f"\nConfusion matrix saved to {output_file}")

if __name__ == "__main__":
    main()
