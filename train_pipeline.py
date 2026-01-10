import torch
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


CRASH_LABELS_PATH = r"e:\Kush\2nd_year\projects\accident_pred\dataset\videos\Crash-1500.txt"
BATCH_SIZE = 4
SHUFFLE = True


crash_filenames = get_image_filenames(CRASH_IMG_FOLDER)
normal_filenames = get_image_filenames(NORMAL_IMG_FOLDER)

crash_labels_1d = read_crash_labels(CRASH_LABELS_PATH)

crash_images_labelled = get_images_labelled(crash_filenames, crash_labels_1d)
normal_images_labelled = get_images_labelled(normal_filenames)


crash_dataset = FrameDataset(CRASH_IMG_FOLDER, crash_images_labelled, transform=None)
normal_dataset = FrameDataset(NORMAL_IMG_FOLDER, normal_images_labelled, transform=None)

crash_loader = DataLoader(crash_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
normal_loader = DataLoader(normal_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)


if __name__ == "__main__":
    print("Testing Crash DataLoader...")
    for images, labels in crash_loader:
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        break

    print("\nTesting Normal DataLoader...")
    for images, labels in normal_loader:
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        break
    
    print("\nDataLoaders ready!")
    print(f"Crash dataset size: {len(crash_dataset)}")
    print(f"Normal dataset size: {len(normal_dataset)}")
