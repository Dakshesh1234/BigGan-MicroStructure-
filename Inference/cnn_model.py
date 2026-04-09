import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension*")
# ======= Dataset =======

class MicrographDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, return_paths=False):
        self.data = pd.read_excel(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.return_paths = return_paths

        self.label_names = sorted(self.data['primary_microconstituent'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}

        valid_rows = []
        for idx, row in self.data.iterrows():
            img_filename = f"Cropped{row['path']}"  # JUST prepend Cropped
            img_path = os.path.join(img_dir, img_filename)
            if os.path.isfile(img_path):
                valid_rows.append((img_path, self.label_to_idx[row['primary_microconstituent']]))
            else:
                print(f"⚠️ Skipping missing image: {img_filename}")

        self.samples = valid_rows

        print("\nLabel Mapping:")
        for label_name, label_idx in self.label_to_idx.items():
            print(f"{label_idx}: {label_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # convert to GRAYSCALE

        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            return image, label, img_path  # For debugging
        else:
            return image, label


# ======= Model =======

class MicrostructureCNN(nn.Module):
    def __init__(self, num_classes):
        super(MicrostructureCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def show_sample_images(dataset, label_mapping, num_images=8):
    idx_to_label = {v: k for k, v in label_mapping.items()}
    
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

    shown = 0
    idx = 0
    while shown < num_images and idx < len(dataset):
        image, label = dataset[idx]
        idx += 1

        # Skip dummy samples
        if label == -1:
            continue

        # Skip empty images (all pixels same)
        if image.max() == image.min():
            continue

        # Image looks good
        image = image.squeeze(0)
        axes[shown].imshow(image, cmap='gray')
        axes[shown].set_title(idx_to_label[label])
        axes[shown].axis('off')
        shown += 1
  

    plt.tight_layout()
    plt.show()

# def show_sample_images(dataset, label_to_idx):
#     import matplotlib.pyplot as plt

#     # show 5 samples from each class
#     samples_per_class = 5
#     shown = {label: 0 for label in label_to_idx.values()}

#     plt.figure(figsize=(15, 10))
#     idx = 1
#     for i in range(len(dataset)):
#         try:
#             image, label, _ = dataset[i]  # Expecting 3 values
#         except ValueError:
#             image, label = dataset[i]     # Fallback if only 2 are returned

#         if shown[label] < samples_per_class:
#             plt.subplot(len(label_to_idx), samples_per_class, idx)
#             plt.imshow(image.squeeze(), cmap='gray')
#             plt.title(f"{label}")
#             plt.axis('off')
#             shown[label] += 1
#             idx += 1
#         if idx > len(label_to_idx) * samples_per_class:
#             break

#     plt.tight_layout()
#     plt.show()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            mask = labels != -1
            images = images[mask]
            labels = labels[mask]
            if len(labels) == 0:
                continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Debugging: check if images match correct labels
def verify_sample_labels(dataset_subset, full_dataset, label_to_idx, num_samples=10):
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    print("\n🔍 Verifying label-image mappings:")
    for i in range(num_samples):
        img, label_idx = dataset_subset[i][:2]
        true_idx = dataset_subset.indices[i]  # index in original dataset
        _, correct_label, path = full_dataset[true_idx]  # get actual info
        label_name = idx_to_label[label_idx]
        correct_name = idx_to_label[correct_label]
        print(f"[{i}] File: {os.path.basename(path)} | Assigned: {label_name} ({label_idx}) | True: {correct_name} ({correct_label})")

# ======= Main function =======

def main():
    # Paths
    csv_file =r'C:\Users\ANVESH\OneDrive\Desktop\lbp\input\highcarbon-micrographs\new_metadata.xlsx'
    img_dir = r'C:\Users\ANVESH\OneDrive\Desktop\lbp\input\highcarbon-micrographs\For Training\Cropped'
    save_path = 'microstructure_try2_model.pth'
    
    # Transform
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

    
    # Dataset and Dataloader
    dataset = MicrographDataset(csv_file=csv_file, img_dir=img_dir, transform=transform,return_paths=False)

    print(f"Classes found: {dataset.label_to_idx}") # <<<< ADD THIS

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    show_sample_images(train_dataset, dataset.label_to_idx)
    #verify_sample_labels(train_dataset, dataset, dataset.label_to_idx)

    
    # Model setup
    num_classes = len(dataset.label_to_idx)
    model = MicrostructureCNN(num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check first batch
    images, labels = next(iter(train_loader))
    print(f"Sample batch - images shape: {images.shape}, labels: {labels}")
    print(f"Unique labels in batch: {labels.unique()}")

    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Skip samples where label == -1 (invalid)
            mask = labels != -1
            images = images[mask]
            labels = labels[mask]

            if len(labels) == 0:
                continue  # Skip if no valid samples in batch

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            print(f"Model outputs shape: {outputs.shape}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)

    print(f"✅ Final Training Accuracy: {train_acc:.2f}%")
    print(f"✅ Final Test Accuracy: {test_acc:.2f}%")
    
    # Save model and label mapping
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_mapping': dataset.label_to_idx
    }, save_path)

    print(f"Training complete and model saved to {save_path}")

# ======= Entry point =======

if __name__ == "__main__":
    main()
