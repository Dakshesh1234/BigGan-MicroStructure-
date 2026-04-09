import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ======= Custom Dataset =======
class MicrographDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_excel(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.label_names = sorted(self.data['primary_microconstituent'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}

        valid_rows = []
        for idx, row in self.data.iterrows():
            img_filename = f"Cropped{row['path']}"
            img_path = os.path.join(img_dir, img_filename)
            if os.path.isfile(img_path):
                valid_rows.append((img_path, self.label_to_idx[row['primary_microconstituent']]))

        self.samples = valid_rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

# ======= Evaluate =======
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

class MicrostructureResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(MicrostructureResNet, self).__init__()

        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Grayscale
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
from sklearn.metrics import classification_report
def evaluate_with_metrics(model, loader, device, label_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("🔍 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names, digits=4))

    accuracy = 100 * (sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels))
    return accuracy


# ======= Main =======
def main():
    csv_file = r'C:\Users\ANVESH\OneDrive\Desktop\lbp\input\highcarbon-micrographs\new_metadata.xlsx'
    img_dir = r'C:\Users\ANVESH\OneDrive\Desktop\lbp\input\highcarbon-micrographs\For Training\Cropped'
    save_path = 'resnet18_microstructure.pth'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MicrographDataset(csv_file, img_dir, transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ======= Load ResNet18 and modify =======
    model=MicrostructureResNet(num_classes=6)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ======= Training =======
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # ======= Evaluation =======
    train_acc = evaluate(model, train_loader, device)

    print(f"✅ Final Training Accuracy: {train_acc:.2f}%")

    print("\n📊 Evaluation on Test Set:")
    test_acc = evaluate_with_metrics(model, test_loader, device, dataset.label_names)
    print(f"✅ Final Test Accuracy: {test_acc:.2f}%")

    torch.save({
        'model_state_dict': model.state_dict(),
        'label_mapping': dataset.label_to_idx
    }, save_path)

    print(f"✅ Model saved to {save_path}")

if __name__ == '__main__':
    main()
