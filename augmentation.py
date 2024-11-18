import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2

# Define the augmentation pipeline
train_transforms = transforms.Compose([
    transforms.RandomRotation((-7.0, 7.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
])

# Original transform without augmentation
eval_transforms = transforms.Compose([
    transforms.ToTensor(),
])

def print_augmentation_summary():
    """Print a detailed summary of the augmentations being applied"""
    print("\n=== Augmentation Summary ===")
    print("1. Random Rotation:")
    print("   - Range: -7 to +7 degrees")
    print("   - Helps with: Rotation invariance")
    
    print("\n2. Random Affine Translation:")
    print("   - Translation: ±10% in both x and y directions")
    print("   - Helps with: Position invariance")
    
    print("\n3. Random Perspective:")
    print("   - Distortion scale: 0.2")
    print("   - Probability: 50%")
    print("   - Helps with: Viewpoint invariance")
    
    print("\n4. ToTensor:")
    print("   - Converts PIL Image to PyTorch tensor")
    print("   - Normalizes pixel values to [0, 1]")
    print("===========================\n")

def get_mnist_data():
    # Download MNIST dataset
    train_data = datasets.MNIST('../data', train=True, download=True, transform=eval_transforms)
    return train_data

def show_augmented_samples(num_samples=10):
    # Print augmentation summary first
    print_augmentation_summary()
    
    # Get the data
    train_data = get_mnist_data()
    
    print(f"Displaying {num_samples} random samples with their augmented versions...")
    
    # Create figure
    fig = plt.figure(figsize=(20, 4))
    
    # Get random samples
    indices = np.random.randint(0, len(train_data), num_samples)
    
    for idx, sample_idx in enumerate(indices):
        # Get original image
        image, label = train_data[sample_idx]
        
        # Create augmented image
        original_pil = transforms.ToPILImage()(image)
        augmented_tensor = train_transforms(original_pil)
        
        # Plot original image
        ax = plt.subplot(2, num_samples, idx + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Original\nLabel: {label}')
        plt.axis('off')
        
        # Plot augmented image
        ax = plt.subplot(2, num_samples, idx + 1 + num_samples)
        plt.imshow(augmented_tensor.squeeze(), cmap='gray')
        plt.title('Augmented')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_augmented_samples() 