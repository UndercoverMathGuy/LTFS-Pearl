import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess(df, nums, cats):
    scaler = MinMaxScaler()
    encoder = OneHotEncoder()
    num_data = scaler.fit_transform(df[nums])
    cat_data = encoder.fit_transform(df[cats])
    tensors_cat = torch.tensor(cat_data, dtype = torch.float32)
    tensors_num = torch.tensor(num_data, dtype = torch.float32)    
    return scaler, encoder, tensors_num, tensors_cat

class ForwardDiffuse:
    def __init__(self, data_nums, data_cats, timestep, total_time, s=0.008):
        self.timestep = timestep
        self.total_time = total_time
        self.data_nums = data_nums
        self.data_cats = data_cats
        self.s = s
        
    def alpha_ct(self, time):
        alpha_ct = (torch.cos(
            (((time/self.total_time) + self.s) / (1+self.s))*(torch.pi/2)
            ))**2
        return alpha_ct
    
    def alpha_schedule(self):
        schedule = torch.linspace(0, self.total_time, self.timestep)
        alpha_schedule = [self.alpha_ct(t) for t in schedule]
        return alpha_schedule
    
    def forward_diffuse_num(self, time, nums):
        alpha_schedule = self.alpha_schedule()
        epsilon = torch.randn_like(self.data_nums)
        nums_noise = torch.sqrt(alpha_schedule[time]) * nums + torch.sqrt(1-alpha_schedule[time]) * epsilon
        return nums_noise
    
    def forward_diffuse_cats(self, temperature=1.0): 
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.data_cats) + 1e-20) + 1e-20)
        noisy_cats = (self.data_cats + gumbel_noise) / temperature
        cats_soft = F.softmax(noisy_cats, dim=-1)
        return cats_soft
    
def Forward_Diffuse(data_nums, data_cats, timestep, total_time, s=0.008):
    forward_diffuse = ForwardDiffuse(data_nums, data_cats, timestep, total_time, s)
    timesteps_list = range(1, total_time+1, timestep)
    data_size = data_nums.shape[1] + data_cats.shape[1] + 1
    data_tensor = torch.zeros((len(timesteps_list), data_size))
    for t in timesteps_list:
        data_nums_noise = forward_diffuse.forward_diffuse_num(t, data_nums)
        data_cats_noise = forward_diffuse.forward_diffuse_cats()
        data_tensor[t, :-1] = torch.cat((data_nums_noise, data_cats_noise), dim=1)
        data_tensor[t, -1] = t
    return data_tensor, torch.cat((data_nums, data_cats), dim=1)