

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def load_data_by_video(data_dir):
    """Load data grouped by video to prevent leakage"""
    video_groups = defaultdict(lambda: {'paths': [], 'label': None})
    
    # Real faces (label = 0)
    real_dir = os.path.join(data_dir, 'real')
    for img_name in os.listdir(real_dir):
        if img_name.endswith('.jpg'):
            # Extract video ID (everything before _face_)
            video_id = img_name.split('_face_')[0]
            video_groups[f"real_{video_id}"]['paths'].append(
                os.path.join(real_dir, img_name)
            )
            video_groups[f"real_{video_id}"]['label'] = 0
    
    # Fake faces (label = 1)
    fake_dir = os.path.join(data_dir, 'fake')
    for img_name in os.listdir(fake_dir):
        if img_name.endswith('.jpg'):
            video_id = img_name.split('_face_')[0]
            video_groups[f"fake_{video_id}"]['paths'].append(
                os.path.join(fake_dir, img_name)
            )
            video_groups[f"fake_{video_id}"]['label'] = 1
    
    return video_groups

def split_by_video(video_groups):
    """Split data at video level, not image level"""
    video_ids = list(video_groups.keys())
    video_labels = [video_groups[vid]['label'] for vid in video_ids]
    
    # Split videos 70/15/15
    train_vids, temp_vids, train_vid_labels, temp_vid_labels = train_test_split(
        video_ids, video_labels, test_size=0.3, random_state=42, stratify=video_labels
    )
    val_vids, test_vids, _, _ = train_test_split(
        temp_vids, temp_vid_labels, test_size=0.5, random_state=42, stratify=temp_vid_labels
    )
    
    # Collect all images from each video split
    def get_images_from_videos(video_list):
        paths = []
        labels = []
        for vid in video_list:
            paths.extend(video_groups[vid]['paths'])
            labels.extend([video_groups[vid]['label']] * len(video_groups[vid]['paths']))
        return paths, labels
    
    train_paths, train_labels = get_images_from_videos(train_vids)
    val_paths, val_labels = get_images_from_videos(val_vids)
    test_paths, test_labels = get_images_from_videos(test_vids)
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def create_model():
    model = models.efficientnet_b0(pretrained=True)
    
    for param in model.features[:5].parameters():
        param.requires_grad = False
    
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    return model.to(device)

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def main():
    DATA_DIR = './processed_faces'
    
    print("Loading data with video-level splitting...")
    video_groups = load_data_by_video(DATA_DIR)
    print(f"Found {len(video_groups)} unique videos")
    
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_by_video(video_groups)
    
    print(f"\nVideo-level split:")
    print(f"Train: {len(train_paths)} images, Val: {len(val_paths)} images, Test: {len(test_paths)} images")
    print(f"Train real/fake: {train_labels.count(0)}/{train_labels.count(1)}")
    print(f"Val real/fake: {val_labels.count(0)}/{val_labels.count(1)}")
    print(f"Test real/fake: {test_labels.count(0)}/{test_labels.count(1)}")
    
    train_dataset = DeepfakeDataset(train_paths, train_labels, train_transform)
    val_dataset = DeepfakeDataset(val_paths, val_labels, val_transform)
    test_dataset = DeepfakeDataset(test_paths, test_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    best_val_acc = 0
    
    print("\n" + "="*50)
    print("Starting Training (Video-Level Split)")
    print("="*50)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_fixed.pth')
            print(f"✅ Saved new best model! (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*50)
    print("Testing on Test Set")
    print("="*50)
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("\n✅ Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
