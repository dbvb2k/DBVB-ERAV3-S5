import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
from tqdm import tqdm

def evaluate(model, device, test_loader):
    print("\n📊 Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print("✅ Evaluation completed!")
    return accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train():
    print("\n🚀 Initializing training process...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load MNIST dataset
    print("\n📥 Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=1000)
    print("✅ Data loading completed!")
    
    # Initialize model
    print("\n🔧 Initializing model and optimizer...")
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    total_params = count_parameters(model)
    print(f"📐 Total model parameters: {total_params:,}")
    print("✅ Model initialization completed!")
    
    # Train for one epoch
    print("\n🏃 Starting training...")
    model.train()
    pbar = tqdm(train_loader, desc='Training batches')
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        pbar.set_postfix({'loss': f'{running_loss:.4f}'})
    
    print("✅ Training completed!")
    
    # Display model parameters again
    print(f"\n📐 Model parameters: {total_params:,}")
    
    # Evaluate model
    accuracy = evaluate(model, device, test_loader)
    print(f"\n📈 Final Model Accuracy: {accuracy:.2f}%")
    
    # Save model with timestamp and accuracy
    print("\n💾 Saving model...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_mnist_{timestamp}_acc{accuracy:.1f}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved successfully as: {save_path}")
    print("\n🎉 All processes completed successfully!")
    
if __name__ == '__main__':
    train() 