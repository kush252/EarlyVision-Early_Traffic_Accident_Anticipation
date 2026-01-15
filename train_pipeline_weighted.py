import torch
import time
import json
import os
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.preprocessor import (
    CRASH_IMG_FOLDER,
    image_resize_toTensor,
    get_images_labelled,
    FrameDataset,
    read_crash_labels,
    get_image_filenames
)
from src.models.vision_models.cnn_model import SimpleCNN

CRASH_LABELS_PATH = r"e:\Kush\2nd_year\projects\accident_pred\dataset\videos\Crash-1500.txt"
BATCH_SIZE = 4
SHUFFLE = True
NUM_WORKERS = 4 

def fit_model(train_loader, val_loader, model, num_epochs=10, checkpoint_path=None, class_weights=None):
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = next(model.parameters()).device

    start_epoch = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],"confusion_matrix": {"TN":0,"FN":0,"TP":0,"FP":0}
    }

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Found checkpoint at '{checkpoint_path}'. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        
        print(f"Resuming training from Epoch {start_epoch+1}...")
    else:
        print(f"No valid checkpoint found. Starting training from scratch on {device}.")

    for epoch in range(start_epoch, num_epochs):
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
            
            if (i+1) % 1000 == 0:
                 print(f"  Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

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
                for p, l in zip(predicted, labels):
                    if p == 1 and l == 1:
                        history['confusion_matrix']['TP'] += 1
                    elif p == 0 and l == 0:
                        history['confusion_matrix']['TN'] += 1
                    elif p == 1 and l == 0:
                        history['confusion_matrix']['FP'] += 1
                    elif p == 0 and l == 1:
                        history['confusion_matrix']['FN'] += 1

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if checkpoint_path:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)

    return history


if __name__ == "__main__":
    
    crash_filenames = get_image_filenames(CRASH_IMG_FOLDER)
    crash_labels_1d = read_crash_labels(CRASH_LABELS_PATH)
    crash_images_labelled = get_images_labelled(crash_filenames, crash_labels_1d)
    
    random.seed(42)  
    random.shuffle(crash_images_labelled)

    total_samples = len(crash_images_labelled)
    count_1 = sum(1 for _, lbl in crash_images_labelled if lbl == 1)
    count_0 = total_samples - count_1
    
    print(f"Dataset Statistics: Normal={count_0}, Crash={count_1}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_weights = None
    if count_1 > 0 and count_0 > 0:
        w0 = total_samples / (2 * count_0) 
        w1 = total_samples / (2 * count_1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
        print(f"Using Class Weights: Normal={w0:.4f}, Crash={w1:.4f}")
    else:
        print("Warning: Dataset is pure class (all 0 or all 1). Weights disabled.")

    split_ratio = 0.8
    split_idx = int(len(crash_images_labelled) * split_ratio)
    
    train_data = crash_images_labelled[:split_idx]
    val_data = crash_images_labelled[split_idx:]

    print(f"Total Crash Samples: {len(crash_images_labelled)}")
    print(f"Training Samples: {len(train_data)}")
    print(f"Validation Samples: {len(val_data)}")

    train_dataset = FrameDataset(CRASH_IMG_FOLDER, train_data, transform=None)
    val_dataset = FrameDataset(CRASH_IMG_FOLDER, val_data, transform=None)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    model = SimpleCNN(num_classes=2).to(device)

    FINAL_MODEL_PATH = os.path.join("models", "accident_detection_weighted_cnn_model.pth")
    CHECKPOINT_PATH = os.path.join("checkpoint_logs", "weighted_training_checkpoint.pth")
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoint_logs", exist_ok=True)

    try:
        start_time = time.time()
        print("\nStarting training loop...")
        
        history = fit_model(
            train_loader, 
            val_loader, 
            model, 
            num_epochs=10, 
            checkpoint_path=CHECKPOINT_PATH,
            class_weights=class_weights
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTraining finished in {elapsed_time:.2f} seconds")
        
        if len(history['train_acc']) > 0:
            print("\nFinal Results:")
            print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
            print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
        
        try:
             precision=history['confusion_matrix']['TP'] / (history['confusion_matrix']['TP'] + history['confusion_matrix']['FP'])
        except ZeroDivisionError:
             precision = 0.0
             
        try:
             recall=history['confusion_matrix']['TP'] / (history['confusion_matrix']['TP'] + history['confusion_matrix']['FN'])
        except ZeroDivisionError:
             recall = 0.0

        try:
             f1_score=2*(precision*recall)/(precision+recall)
        except ZeroDivisionError:
             f1_score = 0.0

        print(f"Confusion Matrix: {history['confusion_matrix']}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")

        with open("weighted_training_history.txt", "w") as f:
            json.dump(history, f, indent=4)
        print("Training history saved to weighted_training_history.txt")

        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print(f"Final Model weights saved to {FINAL_MODEL_PATH}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        print(f"Progress saved to {CHECKPOINT_PATH}")
