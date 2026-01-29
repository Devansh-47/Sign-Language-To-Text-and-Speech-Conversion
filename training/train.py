import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
import sys
from sklearn.model_selection import train_test_split

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import LandmarkTransformer
from backend.preprocessing import Preprocessor

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class HandDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.preprocessor = Preprocessor(target_frames=30)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load JSON
        with open(self.file_paths[idx], 'r') as f:
            data = json.load(f)
        
        # Data is list of frames. 
        # For the provided dataset (static snapshots), it might be inside a list of objects 
        # where each object is a sample. BUT, inspecting A_landmarks.json, it was:
        # [ { "landmarks": [...], "letter": "A", ... }, ... ]
        # So file_paths needs to be (file_path, index_inside_file). 
        # But wait, passing just file path isn't enough if one file has many samples.
        # Let's refactor __init__ to accept list of actual samples (loaded in memory) OR
        # better, list of (file_path, index).
        
        # To avoid loading all gigantic JSONs in __getitem__, let's pre-load them in the main script 
        # and pass a list of 'data_items' to the dataset. The dataset is small (~3000 samples).
        # It fits in memory easily.
        
        pass 

class InMemoryHandDataset(Dataset):
    def __init__(self, samples, labels, augment=False):
        self.samples = samples # List of List[List[float]] (raw landmarks)
        self.labels = labels
        self.augment = augment
        self.preprocessor = Preprocessor(target_frames=30)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_landmarks = self.samples[idx]
        
        # If augment, we might want to perturb raw_landmarks before preprocessing
        # But Preprocessor handles 'sequence creation' by repeating.
        # We want to inject noise during that repetition if training.
        
        # Let's perform standard preprocessing first
        # We need to simulate temporal sequence from this static frame.
        # The preprocessor.process takes [frame] and repeats it.
        # We can implement augmentation inside the dataset by manually creating the sequence with noise.
        
        if self.augment:
             # Manual augmentation for static->temporal
             # 1. Take the static frame
             base_frame = np.array(raw_landmarks, dtype=np.float32)
             
             # 2. Replicate to 30 frames
             sequence = np.tile(base_frame[np.newaxis, ...], (30, 1, 1)) # (30, 21, 3)
             
             # 3. Add Jitter (Gaussian noise)
             noise = np.random.normal(0, 1.5, sequence.shape) # 1.5 pixel noise
             sequence = sequence + noise
             
             # 4. Convert back to list for Preprocessor (or modify Preprocessor to take numpy)
             # Preprocessor expects List[List[List...]].
             # Actually, Preprocessor.process converts to numpy immediately.
             # But it also does normalization.
             # We can subclass Preprocessor or just use it. 
             # Let's just pass the noisy sequence to preprocessor.
             
             # Convert to list
             input_seq = sequence.tolist()
        else:
            input_seq = [raw_landmarks] # Preprocessor will handle replication

        processed = self.preprocessor.process(input_seq)
        
        # Convert to torch
        x = torch.tensor(processed, dtype=torch.float32)
        # Flatten last dims: (30, 21, 3) -> (30, 63)
        x = x.view(30, -1)
        
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def load_data(data_dir):
    all_samples = []
    all_labels = []
    
    files = glob(os.path.join(data_dir, "*_landmarks.json"))
    for fpath in files:
        with open(fpath, 'r') as f:
            data = json.load(f)
            # data is List[Dict]
            for item in data:
                # Basic validation
                if 'landmarks' in item and len(item['landmarks']) == 21:
                    all_samples.append(item['landmarks'])
                    letter = item['letter']
                    all_labels.append(CLASS_TO_IDX[letter])
                    
    return all_samples, all_labels

def main():
    data_dir = r"C:\Users\krish\HandSpeak.ai\dataset_v2_mobile\landmarks"
    print("Loading data...")
    samples, labels = load_data(data_dir)
    print(f"Loaded {len(samples)} samples.")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(samples, labels, test_size=0.2, stratify=labels, random_state=42)
    
    train_ds = InMemoryHandDataset(X_train, y_train, augment=True)
    val_ds = InMemoryHandDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LandmarkTransformer(num_classes=26)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    epochs = 10
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        train_acc = 100. * correct / total
        
        # Val
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
                
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")
            
    # Save final
    torch.save(model.state_dict(), "model_final.pth")

if __name__ == "__main__":
    main()
