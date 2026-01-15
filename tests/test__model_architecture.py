import torch
import random
import sys
import os
import json
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessor import (
    CRASH_IMG_FOLDER,
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

def fit_model(train_loader, val_loader, model, num_epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = next(model.parameters()).device

    print(f"Starting TEST training on device: {device}")

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

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

import time

if __name__ == "__main__":
    start_time_all = time.time()
    print("--- Running Quick Pipeline Test ---")
    
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
    
    random.seed(42)
    random.shuffle(all_crash_images)
    
    subset_size = min(NUM_TEST_IMAGES, len(all_crash_images))
    print(f"Selecting subset of {subset_size} images for testing...")
    
    subset_data = all_crash_images[:subset_size]
    
    if len(subset_data) == 0:
        print("Error: No data found in subset.")
        sys.exit(1)

    split_ratio = 0.8
    split_idx = int(len(subset_data) * split_ratio)
    
    train_data = subset_data[:split_idx]
    val_data = subset_data[split_idx:]
    
    print(f"Test Train Samples: {len(train_data)}")
    print(f"Test Val Samples: {len(val_data)}")

    train_dataset = FrameDataset(CRASH_IMG_FOLDER, train_data, transform=None)
    val_dataset = FrameDataset(CRASH_IMG_FOLDER, val_data, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SimpleCNN(num_classes=2).to(device)

    print("\nStarting Test Training Loop (2 Epochs)...")
    try:
        fit_model(train_loader, val_loader, model, num_epochs=10)
        print("\nTest finished successfully. Pipeline checks out.")
        
        torch.save(model.state_dict(), "test_model_weights.pth")
        print("Model save test passed (test_model_weights.pth).")
        
        if os.path.exists("test_model_weights.pth"):
             os.remove("test_model_weights.pth")
             print("Cleaned up test artifact.")

    except Exception as e:
        print(f"\nPipeline Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTotal Test Duration: {time.time() - start_time_all:.2f} seconds")
