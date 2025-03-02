import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple model with dropout
class MCDropoutModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MCDropoutModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to perform multiple forward passes and compute mean and variance
def predict_with_uncertainty(model, x, num_samples=100):
    # Set the model to training mode to activate dropout during inference
    model.train()  
    predictions = [model(x) for _ in range(num_samples)]
    predictions = torch.stack(predictions)
    
    # Calculate the mean and variance across the predictions
    mean_prediction = predictions.mean(dim=0)
    prediction_variance = predictions.var(dim=0)
    return mean_prediction, prediction_variance

# Example usage:
if __name__ == '__main__':
    # Dummy input data: 10 samples with 5 features each
    x = torch.randn(10, 5)
    
    # Initialize the model
    model = MCDropoutModel(input_dim=5, hidden_dim=20, output_dim=1, dropout_rate=0.5)
    
    # Perform Monte Carlo predictions
    mean, uncertainty = predict_with_uncertainty(model, x, num_samples=100)
    
    print("Predictions Mean:\n", mean)
    print("Predictions Uncertainty (Variance):\n", uncertainty)
