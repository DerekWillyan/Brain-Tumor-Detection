# train.py

import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Definindo transformações para as imagens
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalização básica
])

# Carregando o dataset
data_dir = 'BrainTumor'
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

# Dividindo em conjunto de treino e validação
train_data, val_data = train_test_split(dataset, test_size=0.2, stratify=dataset.targets)

# Dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Usando uma CNN pré-treinada (ex. ResNet18) para transfer learning
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Ajuste para 2 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Função de treinamento
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validação
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
        
        accuracy = correct / len(val_loader.dataset)
        print(f"Validation Accuracy: {accuracy:.4f}")

# Treinando o modelo
train_model(model, criterion, optimizer, num_epochs=10)

# Salvando o modelo treinado
torch.save(model.state_dict(), 'brain_tumor_classifier.pth')
