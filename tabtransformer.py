import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as func
import torch.optim as optim

def create_dataloader(noisy, added_noise, batch_size = 32):
    dataset = TensorDataset(noisy, added_noise)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class TabTransformer(nn.Module):
    def __init__(self, num_inputs, num_heads = 4, num_layers = 4, hidden_dim = 128, dropout = 0.1):
        super(TabTransformer, self).__init__()
        self.input_dim = num_inputs
        self.embedding = nn.Linear(self.input_dim, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, self.input_dim)
        
    def forward(self, x, timestep):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(0)).squeeze(0)
        noise_pred = self.output_layer(x)
        return noise_pred
    
def train_model(inputs, outputs):
    dataloader = create_dataloader(inputs, outputs)
    input_dim = inputs.shape[1]
    model = TabTransformer(num_inputs = input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            noisy, added_noise = batch
            predicted_noise = model(noisy)
            loss = func.mse_loss(predicted_noise, added_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "tabtransformer.pth")
    print("Model saved as tabtransformer.pth")