import torch
import pytest
from model import MNISTModel, count_parameters
from torchvision import datasets
from torch.utils.data import DataLoader
from augmentation import eval_transforms
import torch.nn.functional as F
import numpy as np

@pytest.fixture
def model():
    return MNISTModel()

def test_parameter_count(model):
    # Test that model has less than 25000 parameters
    total_params = count_parameters(model)
    assert total_params < 25000, f"Model has {total_params} parameters, which exceeds 25000"

def test_model_accuracy():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the test dataset
    test_dataset = datasets.MNIST(
        './data', 
        train=False,
        transform=eval_transforms,
        download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Load the model
    model = MNISTModel().to(device)
    
    # Load the trained model weights
    try:
        model.load_state_dict(torch.load('models/mnist_model.pth', map_location=device))
    except:
        pytest.skip("Trained model weights not found. Skipping accuracy test.")
    
    # Set model to evaluation mode
    model.eval()
    
    # Test the model
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get predictions
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}%")
    
    # Assert accuracy is above 95%
    assert accuracy > 95.0, f"Model accuracy ({accuracy:.2f}%) is below the required 95%"

def test_model_structure(model):
    # Test basic model structure
    assert isinstance(model, MNISTModel)
    assert isinstance(model, torch.nn.Module)

def test_forward_pass(model):
    # Test forward pass with batch size of 1
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)
    
    # Test forward pass with batch size of 32
    x = torch.randn(32, 1, 28, 28)
    output = model(x)
    assert output.shape == (32, 10)

def test_output_probabilities(model):
    # Test that output sums to 1 (softmax property)
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_model_components(model):
    # Test presence of key components
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'conv2')
    assert hasattr(model, 'conv3')
    assert hasattr(model, 'conv4')
    assert hasattr(model, 'conv5')
    
    # Test batch normalization layers
    assert hasattr(model, 'bn1')
    assert hasattr(model, 'bn2')
    assert hasattr(model, 'bn3')
    
    # Test dropout layers
    assert hasattr(model, 'dropout1')
    assert hasattr(model, 'dropout2')
    assert hasattr(model, 'dropout3')
    
    # Test GAP layer
    assert hasattr(model, 'gap')

def test_layer_dimensions(model):
    x = torch.randn(1, 1, 28, 28)
    
    # Test first conv block
    x = model.conv1(x)
    assert x.shape == (1, 16, 26, 26)
    
    # Test second conv block
    x = model.conv2(model.dropout1(model.relu1(model.bn1(x))))
    assert x.shape == (1, 16, 24, 24)
    
    # After first maxpool
    x = model.pool1(model.dropout2(model.relu2(model.bn2(x))))
    assert x.shape == (1, 16, 12, 12)
    
    # After third conv block
    x = model.conv3(x)
    assert x.shape == (1, 32, 10, 10)
    
    # After second maxpool
    x = model.pool2(model.dropout3(model.relu3(model.bn3(x))))
    assert x.shape == (1, 32, 5, 5)
    
    # Final convolutions and GAP
    x = model.conv4(x)
    assert x.shape == (1, 10, 5, 5)
    x = model.conv5(x)
    x = model.gap(x)
    assert x.shape == (1, 10, 1, 1)

def test_dropout_rates(model):
    assert model.dropout1.p == 0.25
    assert model.dropout2.p == 0.25
    assert model.dropout3.p == 0.25

def test_model_training_mode(model):
    # Test training mode affects dropout
    model.train()
    assert model.training == True
    x = torch.randn(100, 1, 28, 28)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)  # Outputs should differ due to dropout
    
    # Test eval mode
    model.eval()
    assert model.training == False
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)  # Outputs should be identical in eval mode 

""" def test_model_gradient_flow():
    # Test if gradients are flowing through the model properly 
    model = MNISTModel()
    model.train()
    
    # Create dummy input
    x = torch.randn(1, 1, 28, 28)
    
    # Forward pass
    output = model(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist and are not zero
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.any(param.grad != 0), f"Zero gradient for {name}" """

def test_model_input_dimensions():
    """Test model behavior with different input dimensions"""
    model = MNISTModel()
    model.eval()
    
    # Test batch size variations
    batch_sizes = [1, 32, 64, 128]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"
    
    # Test invalid input dimensions
    with pytest.raises(RuntimeError):
        x = torch.randn(1, 2, 28, 28)  # Wrong number of channels
        model(x)

def test_model_save_load():
    """Test model state saving and loading"""
    model = MNISTModel()
    
    # Save model state
    torch.save(model.state_dict(), 'temp_model.pth')
    
    # Create new model and load state
    new_model = MNISTModel()
    new_model.load_state_dict(torch.load('temp_model.pth'))
    
    # Compare model parameters
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
    
    # Cleanup
    import os
    os.remove('temp_model.pth')

def test_model_reproducibility():
    """Test model output reproducibility"""
    torch.manual_seed(42)
    model = MNISTModel()
    model.eval()
    
    x = torch.randn(1, 1, 28, 28)
    output1 = model(x)
    output2 = model(x)
    
    assert torch.allclose(output1, output2)

def test_softmax_properties():
    """Test if model output satisfies softmax properties"""
    model = MNISTModel()
    model.eval()
    
    x = torch.randn(10, 1, 28, 28)
    output = model(x)
    
    # Sum should be close to 1
    sums = output.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums))
    
    # Values should be between 0 and 1
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)

def test_model_memory_efficiency():
    """Test model memory usage"""
    model = MNISTModel()
    
    # Get model size in MB
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Model should be less than 1MB
    assert model_size < 1, f"Model size ({model_size:.2f}MB) exceeds 1MB"

def test_batch_norm_behavior():
    """Test BatchNorm behavior in train vs eval modes"""
    model = MNISTModel()
    x = torch.randn(100, 1, 28, 28)
    
    # Training mode
    model.train()
    train_outputs = [model(x) for _ in range(2)]
    assert not torch.allclose(train_outputs[0], train_outputs[1])
    
    # Eval mode
    model.eval()
    eval_outputs = [model(x) for _ in range(2)]
    assert torch.allclose(eval_outputs[0], eval_outputs[1])

def test_dropout_behavior():
    """Test Dropout behavior in train vs eval modes"""
    model = MNISTModel()
    x = torch.randn(100, 1, 28, 28)
    
    # Training mode - outputs should be different
    model.train()
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)
    
    # Eval mode - outputs should be identical
    model.eval()
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)