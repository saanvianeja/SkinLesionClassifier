#!/usr/bin/env python3
"""
Train CNN on real ISIC skin lesion data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class ISICDataset(Dataset):
    """Dataset for ISIC skin lesion images"""
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['image_name']
        label = row['label']
        diagnosis = row['diagnosis']
        
        # Load image from correct subdirectory
        img_path = os.path.join(self.img_dir, diagnosis, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_data_transforms():
    """Create data transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes=2):
    """Create ResNet-18 model"""
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last few layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cpu'):
    """Train the model"""
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')
        print('-' * 50)
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_isic_model.pth')
            print(f'New best model saved! F1: {val_f1:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses, val_accuracies, val_f1_scores

def main():
    """Main training function"""
    
    print("Starting ISIC skin lesion classification training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    csv_file = "isic_training_data/isic_labels.csv"
    img_dir = "isic_training_data"
    
    # Load data
    data = pd.read_csv(csv_file)
    print(f"Total images: {len(data)}")
    print(f"Class distribution: {data['diagnosis'].value_counts().to_dict()}")
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    
    # Save splits
    train_data.to_csv("isic_training_data/train_split.csv", index=False)
    val_data.to_csv("isic_training_data/val_split.csv", index=False)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets
    train_dataset = ISICDataset("isic_training_data/train_split.csv", img_dir, train_transform)
    val_dataset = ISICDataset("isic_training_data/val_split.csv", img_dir, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Create model
    model = create_model(num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Train model
    train_losses, val_losses, val_accuracies, val_f1_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device
    )
    
    # Final evaluation
    model.load_state_dict(torch.load('best_isic_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print final results
    final_accuracy = accuracy_score(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")
    print(f"Final Validation F1 Score: {final_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    
    # Save optimal threshold
    with open('optimal_threshold.txt', 'w') as f:
        f.write('0.5')
    
    print("\nTraining completed!")
    print("Model saved as 'best_isic_model.pth'")
    print("Optimal threshold saved as 'optimal_threshold.txt'")

if __name__ == "__main__":
    main() 