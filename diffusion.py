import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess(df, numericals, categoricals):
    scaler = MinMaxScaler()
    num_data = scaler.fit_transform(df[numericals])
    
    encoder = OneHotEncoder()
    cat_data = encoder.fit_transform(df[categoricals])
    
    processed = np.hstack([num_data, cat_data])
    tensors = torch.tensor(processed, dtype=torch.float32)
    return tensors

class ForwardDiffusion:
    def __init__(self, timesteps, beta_start = 1e-4, beta_end = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        
    def diffuse(self, x, t):
        alpha_t = self.alpha_cumprod[t]
        noise = torch.randn_like(x)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
    
class BackDenoise(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        t_embedding = torch.ones_like(x[:, :1]) * t / self.timesteps
        x = torch.cat([x, t_embedding], dim=1)
        return self.net(x)
    
    def TrainDiffusion(data, model, diffusion, epochs=100, batch_size=64, lr=1e-3):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                t = torch.randint(0, diffusion.timesteps, (batch_size(0),)).long()
                noisy_data, noise = diffusion.diffuse(batch, t)
                
                optimizer.zero_grad()
                predicted_noise = model(noisy_data, t)
                loss = loss_fn(predicted_noise, noise)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")
            
    def Create_Data(model, diffusion, num_samples, input_dim):
        model.eval()
        samples = torch.randn((num_samples, input_dim))
        for t in reversed(range(diffusion.timesteps)):
            alpha_t = diffusion.alpha_cumprod[t]
            beta_t = diffusion.betas[t]
            z = torch.randn_like(samples) if t > 0 else 0
            samples = (1 / torch.sqrt(alpha_t)) * (samples - beta_t / torch.sqrt(1 - alpha_t) * model(samples, t)) + torch.sqrt(beta_t) * z
        return samples