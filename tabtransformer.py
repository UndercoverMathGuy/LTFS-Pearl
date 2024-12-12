import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as func
import torch.optim as optim
from diffusion_pre import ForwardDiffuse

def create_dataloader(noisy, added_noise, timesteps, batch_size = 32):
    indices = torch.arange(noisy.size(0))
    dataset = TensorDataset(noisy, added_noise, timesteps, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

class TabTransformer(nn.Module):
    def __init__(self, num_inputs, num_heads = 4, num_layers = 4, hidden_dim = 128, dropout = 0.1):
        super(TabTransformer, self).__init__()
        self.input_dim = num_inputs
        self.embedding = nn.Linear(self.input_dim, hidden_dim)
        self.time_embedding = nn.Embedding(100000, hidden_dim)  # Add time step embedding
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True  # Set batch_first to True
            ),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, self.input_dim)
        
    def forward(self, x, timesteps):
        x = self.embedding(x)
        time_emb = self.time_embedding(timesteps)
        x = x + time_emb  # Combine input with time embedding
        x = self.transformer(x)
        noise_pred = self.output_layer(x)
        return noise_pred
    
def train_model(inputs, outputs, timesteps, num_steps, total_time, s=0.008):
    # Verify data alignment before training
    forward_diffuse = ForwardDiffuse(
        data_nums=None,  # Placeholder, actual data will be provided per sample
        data_cats=None,
        num_steps=num_steps,
        total_time=total_time,
        s=s
    )
    alpha_schedule = forward_diffuse.alpha_schedule

    for idx in range(len(inputs)):
        noisy_sample = inputs[idx]
        added_noise_sample = outputs[idx]
        timestep = timesteps[idx]

        alpha_t = alpha_schedule[timestep]
        # Reconstruct the original data
        reconstructed_original = (noisy_sample - torch.sqrt(1 - alpha_t) * added_noise_sample) / torch.sqrt(alpha_t)
        # Recompute the noisy data
        expected_noisy_sample = torch.sqrt(alpha_t) * reconstructed_original + torch.sqrt(1 - alpha_t) * added_noise_sample

        # Check data alignment
        if not torch.allclose(expected_noisy_sample, noisy_sample, atol=1e-5):
            print(f"Data misalignment at index {idx}")
        else:
            print(f"Data aligned at index {idx}")

    # Proceed with training
    dataloader = create_dataloader(inputs, outputs, timesteps)
    input_dim = inputs.shape[1]
    model = TabTransformer(num_inputs = input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            noisy, added_noise, batch_timesteps = batch
            predicted_noise = model(noisy, batch_timesteps)
            loss = func.mse_loss(predicted_noise, added_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "tabtransformer.pth")
    print("Model saved as tabtransformer.pth")