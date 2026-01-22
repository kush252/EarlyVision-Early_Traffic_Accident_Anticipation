import torch
import random
import sys
import os
import json
import numpy as np
import time
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessor import (
    CRASH_IMG_FOLDER,
    NORMAL_IMG_FOLDER,
    get_images_labelled,
    FrameDataset,
    read_crash_labels,
    get_image_filenames
)
from src.models.vision_models.cnn_model import SimpleCNN

CRASH_LABELS_PATH = r"e:\Kush\2nd_year\projects\accident_pred\dataset\videos\Crash-1500.txt"
BATCH_SIZE = 4
NUM_WORKERS = 0 
NUM_TEST_IMAGES = 200 

from sklearn.metrics import recall_score, f1_score, confusion_matrix

def fit_model(train_loader, val_loader, model, class_weights=None, num_epochs=2):
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = next(model.parameters()).device

    print(f"Starting TEST training on device: {device}")

    val_preds = []
    val_targets = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        val_recall = recall_score(val_targets, val_preds, pos_label=1, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, pos_label=1, zero_division=0)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    return val_targets, val_preds

if __name__ == "__main__":
    start_time_all = time.time()
    print("--- Running Quick Weighted Pipeline Test (Crash Only) ---")
    
    print("Loading filenames...")
    try:
        crash_filenames = get_image_filenames(CRASH_IMG_FOLDER)
        print(f"Found {len(crash_filenames)} total crash images.")
    except Exception as e:
        print(f"Error loading filenames: {e}")
        sys.exit(1)
    
    print("Reading labels...")
    try:
        crash_labels_1d = read_crash_labels(CRASH_LABELS_PATH)
    except Exception as e:
        print(f"Error reading labels: {e}")
        sys.exit(1)
    
    print("Pairing images and labels...")
    all_crash_images = get_images_labelled(crash_filenames, crash_labels_1d)
    
    accident_frames = [x for x in all_crash_images if x[1] == 1]
    normal_frames = [x for x in all_crash_images if x[1] == 0]
    
    print(f"Total available Accident frames in Crash Folder: {len(accident_frames)}")
    print(f"Total available Normal frames in Crash Folder: {len(normal_frames)}")
    
    n_accident = min(len(accident_frames), NUM_TEST_IMAGES // 2)
    n_normal = NUM_TEST_IMAGES - n_accident
    
    subset_data = accident_frames[:n_accident] + normal_frames[:n_normal]
    random.shuffle(subset_data)
    
    count_1 = sum(1 for _, lbl in subset_data if lbl == 1)
    count_0 = len(subset_data) - count_1
    
    print(f"Subset Statistics: Normal={count_0}, Crash={count_1}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if count_1 > 0 and count_0 > 0:
        total = len(subset_data)
        w0 = total / (2 * count_0) 
        w1 = total / (2 * count_1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
        print(f"Class Weights: Normal={w0:.4f}, Crash={w1:.4f}")
    else:
        class_weights = None
        print("Warning: One class missing in subset, weights disabled.")

    split_idx = int(len(subset_data) * 0.8)
    
    train_data = subset_data[:split_idx]
    val_data = subset_data[split_idx:]
    
    print(f"Test Train Samples: {len(train_data)}")
    print(f"Test Val Samples: {len(val_data)}")

    ds_train = FrameDataset(CRASH_IMG_FOLDER, train_data, transform=None)
    ds_val = FrameDataset(CRASH_IMG_FOLDER, val_data, transform=None)
    
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"Using device: {device}")
    model = SimpleCNN(num_classes=2).to(device)

    print("\nStarting Weighted Test Training Loop...")
    try:
        y_true, y_pred = fit_model(train_loader, val_loader, model, class_weights=class_weights, num_epochs=10)
        print("\nTest finished successfully.")
        
        print("\nConfusion Matrix (Rows=True, Cols=Pred):")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print(" [TN, FP]")
        print(" [FN, TP]")

        torch.save(model.state_dict(), "test_weighted_model.pth")
        print("Model save test passed.")
        
        time.sleep(1)
        if os.path.exists("test_weighted_model.pth"):
             os.remove("test_weighted_model.pth")
             print("Cleaned up test artifact.")

    except Exception as e:
        print(f"\nWeighted Pipeline Test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTotal Test Duration: {time.time() - start_time_all:.2f} seconds")
