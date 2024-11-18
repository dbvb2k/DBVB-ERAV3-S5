import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First Block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # 28x28 -> 26x26
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        
        # Second Block
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)  # 26x26 -> 24x24
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        
        # First MaxPool
        self.pool1 = nn.MaxPool2d(2, 2)  # 24x24 -> 12x12
        
        # Third Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)  # 12x12 -> 10x10
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.25)
        
        # Second MaxPool
        self.pool2 = nn.MaxPool2d(2, 2)  # 10x10 -> 5x5
        
        # Final Convolution Blocks
        self.conv4 = nn.Conv2d(32, 10, kernel_size=1)  # 5x5 -> 5x5
        self.conv5 = nn.Conv2d(10, 10, kernel_size=5)  # 5x5 -> 1x1
        self.bn4 = nn.BatchNorm2d(10)
        self.relu4 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # First MaxPool
        x = self.pool1(x)
        
        # Third Block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Second MaxPool
        x = self.pool2(x)
        
        # Final Convolutions
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # Output
        x = self.flatten(x)
        x = self.softmax(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = MNISTModel()
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}") 