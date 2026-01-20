import torch
import torch.nn as nn
import os
import sys
import time
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessor import image_resize_toTensor
from src.data.preprocessor import read_crash_labels
from src.models.vision_models.cnn_extractor import CNNExtractor
from src.utils.lstm_utils import frames_to_sequence,framelabel_to_sequencelabel

from sklearn.model_selection import train_test_split
from src.models.temporal_model.LSTM_model import RiskLSTM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "accident_detection_weighted_cnn_model.pth")

output_pth = os.path.join(PROJECT_ROOT, "dataset", "frames","Crash-1500")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


L=10
K=5

FEATURE_DIM = 512
HIDDEN_DIM = 256
NUM_LAYERS = 2
BIDIRECTIONAL = False
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5


LSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "risk_lstm_model.pth")
LSTM_HISTORY_PATH = os.path.join(PROJECT_ROOT, "models", "lstm_training_history.txt")


def create_batches(X, y, batch_size):
    """Manually create batches from data."""
    n_samples = len(X)
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_x = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        batches.append((batch_x, batch_y))
    return batches


def shuffle_data(X, y):
    """Shuffle data for training."""
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def train_one_epoch(model, X_train, y_train, criterion, optimizer, device, batch_size):
    """Train for one epoch using manual batching."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    

    X_shuffled, y_shuffled = shuffle_data(X_train, y_train)
    

    batches = create_batches(X_shuffled, y_shuffled, batch_size)
    
    for batch_x, batch_y in batches:
        batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
        batch_y = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        
    
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, X_val, y_val, criterion, device, batch_size):
    """Validate the model using manual batching."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    

    batches = create_batches(X_val, y_val, batch_size)
    
    with torch.no_grad():
        for batch_x, batch_y in batches:
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(1).to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def train_lstm(sequences,y):
    print("\n" + "="*60)
    print("STARTING LSTM TRAINING")
    print("="*60)
    
    X = np.array(sequences, dtype=np.float32)
    y_labels = np.array(y, dtype=np.float32)
    
    print(f"\nDataset Info:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Sequence shape: {X.shape}")
    print(f"  Labels shape: {y_labels.shape}")
    print(f"  Positive samples: {np.sum(y_labels)} ({100*np.sum(y_labels)/len(y_labels):.2f}%)")
    print(f"  Negative samples: {len(y_labels) - np.sum(y_labels)} ({100*(len(y_labels)-np.sum(y_labels))/len(y_labels):.2f}%)")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"\nTrain/Val Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    n_train_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    n_val_batches = (len(X_val) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Training batches per epoch: {n_train_batches}")
    print(f"  Validation batches: {n_val_batches}")
    
    model = RiskLSTM(
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL
    ).to(device)
    
    
    pos_weight = torch.tensor([(len(y_labels) - np.sum(y_labels)) / max(np.sum(y_labels), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Device: {device}")
    print(f"  Pos weight: {pos_weight.item():.4f}")
    
    print("\n" + "-"*60)
    print("Starting training...")
    print("-"*60)
    
    training_start = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
       
        train_loss, train_acc = train_one_epoch(model, X_train, y_train, criterion, optimizer, device, BATCH_SIZE)
        
        val_loss, val_acc, _, _ = validate(model, X_val, y_val, criterion, device, BATCH_SIZE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
    total_training_time = time.time() - training_start
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    
    if len(history['train_acc']) > 0:
        print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")

    with open(LSTM_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {LSTM_HISTORY_PATH}")
    

    torch.save(model.state_dict(), LSTM_MODEL_PATH)
    print(f"Model saved to: {LSTM_MODEL_PATH}")
    
    return model, history


if __name__ == "__main__" or True:  
    
    crash_labels_path=os.path.join(PROJECT_ROOT, "dataset", "videos", "Crash-1500.txt")
    print(f"Extracting features from frames in: {output_pth}...")
    print(f"Using device: {device}")

    extractor = CNNExtractor(WEIGHTS_PATH).to(device)
    extractor.eval()

    feature_list = []
    frame_files = sorted([f for f in os.listdir(output_pth) if f.endswith('.jpg')])
    total_frames = len(frame_files)

    if not frame_files:
        print("No frames found to process!")
    else:
        print(f"Total frames to process: {total_frames}")
        start_time = time.time()
        
        with torch.no_grad():
            for i, frame_file in enumerate(frame_files):
                img_tensor = image_resize_toTensor(output_pth, frame_file)
                img_tensor = img_tensor.unsqueeze(0).to(device)
            
                features = extractor(img_tensor)
                features_np = features.cpu().numpy().flatten()
                feature_list.append(features_np)
                
                if (i + 1) % 5000 == 0 or (i + 1) == total_frames:
                    elapsed = time.time() - start_time
                    speed = (i + 1) / elapsed
                    remaining = (total_frames - i - 1) / speed if speed > 0 else 0
                    print(f"Progress: {i + 1}/{total_frames} ({100*(i+1)/total_frames:.1f}%) | "
                        f"Elapsed: {elapsed:.1f}s | Speed: {speed:.1f} frames/s | ETA: {remaining:.1f}s")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nFeature extraction complete!")
        print(f"Total features extracted: {len(feature_list)}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")


    sequences=frames_to_sequence(feature_list,L,K)
    labels=read_crash_labels(crash_labels_path)
    y=framelabel_to_sequencelabel(labels[:50],L,K)
 
    trained_model, training_history = train_lstm(sequences,y)
