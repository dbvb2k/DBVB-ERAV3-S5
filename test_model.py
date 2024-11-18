import torch
import pytest
from model import MNISTModel, count_parameters

@pytest.fixture
def model():
    return MNISTModel()

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

def test_parameter_count(model):
    # Test that model has less than 25000 parameters
    total_params = count_parameters(model)
    assert total_params < 25000, f"Model has {total_params} parameters, which exceeds 25000"

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