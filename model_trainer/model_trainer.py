import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

class ModelTrainer:
    def __init__(self, model, file_path, file_type, val_loader=None, device=None, lr=0.001, epochs=10):
        self.model = model
        self.file_path = file_path
        self.file_type = file_type
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.train_loader = self.load_data(file_path, file_type)

    def load_data(self, file_path, file_type):
        if file_type == 'csv':
            # Implement CSV loading logic here
            print(f"Loading data from CSV: {file_path}")
            return DataLoader([])  # Placeholder
        elif file_type == 'yolo':
            # Implement YOLO format loading
            print(f"Loading data from YOLO: {file_path}")
            return DataLoader([])  # Placeholder
        elif file_type == 'coco':
            # Implement COCO format loading
            print(f"Loading data from COCO: {file_path}")
            return DataLoader([])  # Placeholder
        else:
            raise ValueError("Unsupported file type")

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        return accuracy

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            if self.val_loader:
                val_accuracy = self.validate()
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
