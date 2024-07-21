import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiscaleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MultiscaleCNN, self).__init__()
        
        # Stream for original scale
        self.conv1_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Stream for half scale
        self.conv2_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Stream for quarter scale
        self.conv3_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Combined processing
        self.conv_combined = nn.Conv2d(64*3, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Original scale
        x1 = F.relu(self.conv1_1(x))
        x1 = F.max_pool2d(x1, 2)
        x1 = F.relu(self.conv1_2(x1))
        
        # Half scale
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x2 = F.relu(self.conv2_1(x2))
        x2 = F.relu(self.conv2_2(x2))
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        # Quarter scale
        x3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x3 = F.relu(self.conv3_1(x3))
        x3 = F.relu(self.conv3_2(x3))
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        # Combine features
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Further processing
        x = F.relu(self.conv_combined(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Usage example
input_channels = 3  # For RGB images
num_classes = 10  # Example number of classes
model = MultiscaleCNN(input_channels, num_classes)

# Example forward pass
batch_size = 4
input_tensor = torch.randn(batch_size, input_channels, 224, 224)
output = model(input_tensor)
print(output.shape)  # Should be [batch_size, num_classes]
