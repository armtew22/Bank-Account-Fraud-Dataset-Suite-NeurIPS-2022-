import torch
import torch.nn as nn

class FraudClassifier(nn.Module):
    def __init__(self):
        """
        Creates the model for a binary classifier given 30 features
    
        Returns:
            A PyTorch sequental neural network
        """
        super(FraudClassifier, self).__init__()
        
        self.layer1 = nn.Linear(30, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) # New to dropout, not sure where the rate should be
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.layer3(x))
        return x

model = FraudClassifier()

