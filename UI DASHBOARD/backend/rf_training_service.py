#!/usr/bin/env python3
"""
IMMEDIATE WORKING PREDICTION SCRIPT
This will work with your exact setup RIGHT NOW
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import joblib

class ViralGenomePredictionSystem:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
    def create_kmers(self, sequence, k=3):
        """Create k-mers from sequence"""
        return [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    
    def extract_features(self, sequences):
        """Extract features from sequences"""
        kmer_sequences = []
        for seq in sequences:
            # Clean sequence - only keep A, T, G, C
            clean_seq = ''.join([base.upper() for base in seq if base.upper() in 'ATGC'])
            if len(clean_seq) < 3:  # Skip very short sequences
                clean_seq = "ATGC" * 10  # Add dummy sequence
            
            kmers = self.create_kmers(clean_seq, k=3)
            kmer_sequences.append(' '.join(kmers))
        
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(max_features=5000, token_pattern=r'\b\w+\b')
            X = self.vectorizer.fit_transform(kmer_sequences)
        else:
            X = self.vectorizer.transform(kmer_sequences)
            
        return X.toarray()
    
    def train_model(self, csv_file):
        """Train the model on your CSV data"""
        print(f"🎯 Training model with: {csv_file}")
        
        try:
            # Load your CSV
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(df)} sequences")
            
            # Get columns - handle different column names
            columns = df.columns.tolist()
            print(f"📊 Columns found: {columns}")
            
            # Extract sequences and labels
            sequences = df.iloc[:, 1].astype(str).tolist()  # Second column (sequence)
            labels = df.iloc[:, 2].astype(int).tolist()     # Third column (label)
            
            print(f"📈 Data stats:")
            print(f"   Sequences: {len(sequences)}")
            print(f"   Viral (1): {sum(labels)}")
            print(f"   Non-viral (0): {len(labels) - sum(labels)}")
            
            # Extract features
            print("🧬 Extracting features...")
            X = self.extract_features(sequences)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            print("🤖 Training Random Forest...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=20,
                min_samples_split=5
            )
            
            self.model.fit(X_train, y_train)
            
            # Test accuracy
            train_acc = self.model.score(X_train, y_train)
            test_acc = self.model.score(X_test, y_test)
            
            print(f"✅ Training completed!")
            print(f"   🎯 Train accuracy: {train_acc:.4f}")
            print(f"   🎯 Test accuracy: {test_acc:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def predict_sequences(self, sequences):
        """Predict viral/non-viral for sequences"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None
        
        try:
            # Extract features
            X = self.extract_features(sequences)
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            results = []
            for i, (seq, pred, prob) in enumerate(zip(sequences, predictions, probabilities)):
                results.append({
                    'sequence_id': f'seq_{i+1}',
                    'sequence': seq[:50] + "..." if len(seq) > 50 else seq,
                    'prediction': int(pred),
                    'label': "🦠 Viral" if pred == 1 else "✅ Non-Viral",
                    'confidence': float(max(prob)),
                    'viral_probability': float(prob[1]),
                    'nonviral_probability': float(prob[0])
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
    
    def save_model(self, filepath):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'is_trained': self.is_trained
            }
            joblib.dump(data, filepath)
            print(f"💾 Model saved: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.is_trained = data['is_trained']
            print(f"✅ Model loaded: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return False

def load_fasta_sequences(fasta_file):
    """Load sequences from FASTA file"""
    sequences = []
    seq_ids = []
    current_seq = ""
    current_id = ""
    
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        seq_ids.append(current_id)
                    current_id = line[1:]  # Remove '>'
                    current_seq = ""
                else:
                    current_seq += line
            
            # Add last sequence
            if current_seq:
                sequences.append(current_seq)
                seq_ids.append(current_id)
        
        print(f"✅ Loaded {len(sequences)} sequences from FASTA")
        return sequences, seq_ids
        
    except Exception as e:
        print(f"❌ Error loading FASTA: {e}")
        return [], []

def create_test_fasta():
    """Create a test FASTA file"""
    test_file = "test_sequences.fasta"
    
    test_data = [
        ">test_sequence_1",
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        ">test_sequence_2",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA", 
        ">mystery_sequence_3",
        "TTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAA",
        ">viral_like_sequence",
        "CCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAA",
        ">random_sequence_5",
        "ATGCGATGCGATGCGATGCGATGCGATGCGATGCGATGCGATGCG"
    ]
    
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_data))
    
    print(f"✅ Created test file: {test_file}")
    return test_file

def run_complete_workflow():
    """Run the complete workflow RIGHT NOW"""
    print("🚀 VIRAL GENOME PREDICTION - COMPLETE WORKFLOW")
    print("=" * 60)
    
    # Initialize system
    predictor = ViralGenomePredictionSystem()
    
    # Step 1: Check if model exists
    model_file = "backend/models/trained_model.pkl"
    
    if os.path.exists(model_file):
        print(f"📁 Found existing model: {model_file}")
        if predictor.load_model(model_file):
            print("✅ Model loaded successfully!")
        else:
            print("❌ Model load failed, will retrain...")
            predictor.is_trained = False
    
    # Step 2: Train if needed
    if not predictor.is_trained:
        csv_file = "fullset_train.csv"
        if os.path.exists(csv_file):
            print(f"🎯 Training with: {csv_file}")
            if predictor.train_model(csv_file):
                predictor.save_model(model_file)
            else:
                print("❌ Training failed!")
                return
        else:
            print(f"❌ CSV file not found: {csv_file}")
            return
    
    # Step 3: Create test data if needed
    test_file = "test_sequences.fasta"
    if not os.path.exists(test_file):
        test_file = create_test_fasta()
    
    # Step 4: Load test sequences
    sequences, seq_ids = load_fasta_sequences(test_file)
    
    if not sequences:
        print("❌ No test sequences found!")
        return
    
    # Step 5: Make predictions
    print(f"\n🔮 Making predictions on {len(sequences)} sequences...")
    results = predictor.predict_sequences(sequences)
    
    if not results:
        print("❌ Prediction failed!")
        return
    
    # Step 6: Display results
    print("\n" + "=" * 80)
    print("🧬 PREDICTION RESULTS")
    print("=" * 80)
    print(f"{'ID':<20} {'Prediction':<15} {'Confidence':<12} {'Viral %':<10} {'Sequence':<30}")
    print("-" * 80)
    
    viral_count = 0
    for result in results:
        if result['prediction'] == 1:
            viral_count += 1
        
        print(f"{result['sequence_id']:<20} "
              f"{result['label']:<15} "
              f"{result['confidence']*100:.1f}%{'':<7} "
              f"{result['viral_probability']*100:.1f}%{'':<6} "
              f"{result['sequence']:<30}")
    
    # Summary
    total = len(results)
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   Total sequences: {total}")
    print(f"   Predicted Viral: {viral_count} ({viral_count/total*100:.1f}%)")
    print(f"   Predicted Non-Viral: {total-viral_count} ({(total-viral_count)/total*100:.1f}%)")
    print("✅ Predictions completed successfully!")

def quick_test_with_manual_sequences():
    """Quick test with manual DNA sequences"""
    print("🧪 QUICK TEST WITH MANUAL SEQUENCES")
    print("=" * 40)
    
    # Initialize system
    predictor = ViralGenomePredictionSystem()
    
    # Load or train model
    model_file = "backend/models/trained_model.pkl"
    
    if os.path.exists(model_file):
        if not predictor.load_model(model_file):
            print("❌ Model load failed!")
            return
    else:
        # Train quickly
        csv_file = "fullset_train.csv"
        if predictor.train_model(csv_file):
            predictor.save_model(model_file)
        else:
            print("❌ Training failed!")
            return
    
    # Test sequences
    test_sequences = [
        "ATCGATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "TTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCC",
        "CCCGGGAAATTTCCCGGGAAATTTCCCGGGAAA"
    ]
    
    print(f"🔮 Testing {len(test_sequences)} sequences...")
    results = predictor.predict_sequences(test_sequences)
    
    if results:
        print("\n📊 RESULTS:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['label']} (confidence: {result['confidence']*100:.1f}%)")
        print("✅ Test completed!")
    else:
        print("❌ Test failed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "quick":
            quick_test_with_manual_sequences()
        elif command == "full":
            run_complete_workflow()
        else:
            print("Usage:")
            print("  python script.py quick  - Quick test")
            print("  python script.py full   - Full workflow")
    else:
        # Default: run full workflow
        run_complete_workflow()

print("\n🎯 READY COMMANDS:")
print("python immediate_prediction_fix.py        # Full workflow")
print("python immediate_prediction_fix.py quick  # Quick test")
print("\nYour prediction system is ready! 🚀")