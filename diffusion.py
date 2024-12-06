from diffusion_pre import preprocess, ForwardDiffusion, BackDenoise, TrainDiffusion, CreateData
import datetime
import torch

def train_diffusion(df, timesteps, nums, cats):
    data = preprocess(df, nums, cats)
    input_dim = data.shape[1]
    diffusion = ForwardDiffusion(timesteps)
    model = BackDenoise(input_dim)
    
    TrainDiffusion(data, model, diffusion, epochs = 1000)
    
    
def save_model(model, file_prefix="diffusion_model"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"{file_prefix}_{timestamp}.pth"
    
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
    return file_path

def load_model(model_class, file_path):
    model = model_class()
    model.load_state_dict(torch.load(file_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Denoising model loaded from {file_path}")
    return model