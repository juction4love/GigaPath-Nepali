import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import openslide
import numpy as np
import os

# Custom Pathology Dataset (NHRC slides)
class NHRCPathologyDataset(Dataset):
    def __init__(self, slide_paths, labels, transform=None):
        self.slide_paths = slide_paths
        self.labels = labels  # 0: Benign, 1: Malignant
        self.transform = transform
    
    def __len__(self):
        return len(self.slide_paths)
    
    def __getitem__(self, idx):
        slide_path = self.slide_paths[idx]
        label = self.labels[idx]
        
        # Extract 1024x1024 patch from center (WSI)
        slide = openslide.OpenSlide(slide_path)
        patch = slide.read_region((10000, 10000), 0, (1024, 1024))
        patch = patch.convert('RGB')
        
        if self.transform:
            patch = self.transform(patch)
        
        return patch, torch.tensor(label, dtype=torch.long)

# Data transforms (Pathology specific)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ResNet50 Model (Pretrained + Fine-tune)
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary: Benign/Malignant

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy data (NHRC slides replace)
slide_paths = ['data/nhrc_slide_001.svs', 'data/nhrc_slide_002.svs']
labels = [0, 1]  # Benign, Malignant
dataset = NHRCPathologyDataset(slide_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop (Capstone prototype)
for epoch in range(10):
    model.train()
    running_loss = 0.0
    
    for patches, targets in dataloader:
        patches, targets = patches.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}')

# Save model (Arkansas thesis ready)
torch.save(model.state_dict(), 'models/gigapath_resnet50.pth')
print("GigaPath-Nepali baseline trained! 85% accuracy target")
