import torch
import time
import os
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.preprocessor import (
    CRASH_IMG_FOLDER,
    NORMAL_IMG_FOLDER,
    image_resize_toTensor,
    get_images_labelled,
    FrameDataset,
    read_crash_labels,
    get_image_filenames
)
from src.models.cnn_model import SimpleCNN


CRASH_LABELS_PATH = r"e:\Kush\2nd_year\projects\accident_pred\dataset\videos\Crash-1500.txt"
BATCH_SIZE = 4
SHUFFLE = True
NUM_WORKERS = 4  # Enable parallel data loading


def fit_model(loader,model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10  # start with 10–30

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        device = next(model.parameters()).device

        total_steps = len(loader)
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    crash_filenames = get_image_filenames(CRASH_IMG_FOLDER)
    normal_filenames = get_image_filenames(NORMAL_IMG_FOLDER)

    crash_labels_1d = read_crash_labels(CRASH_LABELS_PATH)

    crash_images_labelled = get_images_labelled(crash_filenames, crash_labels_1d)
    normal_images_labelled = get_images_labelled(normal_filenames)

    crash_dataset = FrameDataset(CRASH_IMG_FOLDER, crash_images_labelled, transform=None)
    normal_dataset = FrameDataset(NORMAL_IMG_FOLDER, normal_images_labelled, transform=None)

    # Use NUM_WORKERS for parallel loading, pin_memory for faster GPU transfer
    crash_loader = DataLoader(
        crash_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    normal_loader = DataLoader(
        normal_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=2).to(device)

    MODEL_PATH = "accident_detection_model.pth"
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model weights at '{MODEL_PATH}'. Loading...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Resuming training from saved weights.")
    else:
        print("No saved weights found. Starting training from scratch.")

    try:
        start_time = time.time()
        fit_model(crash_loader, model)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"Training finished in {elapsed_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")

    # Save weights regardless of whether it finished or was interrupted
    torch.save(model.state_dict(), "accident_detection_model.pth")
    print("Model weights saved to accident_detection_model.pth")

    # Try to print metrics if they exist, but don't crash if they don't
    try:
        print(f"model accuracy: {model.accuracy}")
        print(f"model loss: {model.loss}")
        print(f"model precision: {model.precision}")
        print(f"model recall: {model.recall}")
        print(f"model f1: {model.f1}")
    except AttributeError:
        print("Metrics retrieval skipped (attributes like .accuracy not set on model).")

# if __name__ == "__main__":
#     print("Testing Crash DataLoader...")
#     for images, labels in crash_loader:
#         print(f"  Images shape: {images.shape}")
#         print(f"  Labels shape: {labels.shape}")
#         print(f"  Labels: {labels}")
#         break

#     print("\nTesting Normal DataLoader...")
#     for images, labels in normal_loader:
#         print(f"  Images shape: {images.shape}")
#         print(f"  Labels shape: {labels.shape}")
#         print(f"  Labels: {labels}")
#         break
    
#     print("\nDataLoaders ready!")
#     print(f"Crash dataset size: {len(crash_dataset)}")
#     print(f"Normal dataset size: {len(normal_dataset)}")
