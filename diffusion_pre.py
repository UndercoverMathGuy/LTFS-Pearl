import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess(df, nums, cats):
    scaler = MinMaxScaler()
    encoder = OneHotEncoder()
    num_data = scaler.fit_transform(df[nums])
    cat_data = encoder.fit_transform(df[cats]).toarray()  # Ensure it's a 2D array
    tensors_cat = torch.tensor(cat_data, dtype=torch.float32)
    tensors_num = torch.tensor(num_data, dtype=torch.float32)
    return scaler, encoder, tensors_num, tensors_cat

class ForwardDiffuse:
    def __init__(self, data_nums, data_cats, timestep, total_time, s=0.008):
        self.timestep = timestep
        self.total_time = total_time
        self.data_nums = data_nums
        self.data_cats = data_cats
        self.s = s
        self.alpha_schedule = self._compute_alpha_schedule()  # Precompute schedule
    
    def _alpha_ct(self, time):
        return (torch.cos((((time / self.total_time) + self.s) / (1 + self.s)) * (torch.pi / 2)))**2
    
    def _compute_alpha_schedule(self):
        schedule = torch.linspace(0, self.total_time, self.timestep)
        return torch.tensor([self._alpha_ct(t) for t in schedule])
    
    def forward_diffuse_num(self, time, nums):
        alpha_t = self.alpha_schedule[time]
        epsilon = torch.randn_like(nums)
        nums_noise = torch.sqrt(alpha_t) * nums + torch.sqrt(1 - alpha_t) * epsilon
        return nums_noise
    
    def forward_diffuse_cats(self, time, cats):
        alpha_t = self.alpha_schedule[time]
        epsilon = torch.randn_like(cats)
        cats_noise = torch.sqrt(alpha_t) * cats + torch.sqrt(1 - alpha_t) * epsilon
        return cats_noise

def Forward_Diffuse(data_nums, data_cats, timestep, total_time, s=0.008):
    forward_diffuse = ForwardDiffuse(data_nums, data_cats, timestep, total_time, s)
    timesteps_list = list(range(0, total_time, timestep))
    batch_size = data_nums.shape[0]
    data_size = data_nums.shape[1] + data_cats.shape[1] + 1
    data_tensor = torch.zeros((len(timesteps_list), batch_size, data_size))  # Include batch dimension
    
    for idx, t in enumerate(timesteps_list):
        data_nums_noise = forward_diffuse.forward_diffuse_num(idx, data_nums)
        data_cats_noise = forward_diffuse.forward_diffuse_cats(idx, data_cats)
        data_tensor[idx, :, :-1] = torch.cat((data_nums_noise, data_cats_noise), dim=1)
        data_tensor[idx, :, -1] = t  # Assign timestep
    
    return data_tensor, torch.cat((data_nums, data_cats), dim=1)

class DeNoiseData:
    def __init__(self, data_nums_cl, data_cats_cl, data_nums_n, data_cats_n, timestep, total_time, s=0.008):
        self.data_nums = data_nums_cl
        self.data_cats = data_cats_cl
        self.timestep = timestep
        self.total_time = total_time
        self.s = s
        self.data_numnoise = data_nums_n
        self.data_catnoise = data_cats_n
    
    def alpha_ct(self, time):
        return (torch.cos((((time / self.total_time) + self.s) / (1 + self.s)) * (torch.pi / 2)))**2
    
    def noise_added_num(self, time):
        alpha_t = self.alpha_ct(time)
        noise_num = (self.data_numnoise - (torch.sqrt(alpha_t)*self.data_nums))/(torch.sqrt(1-alpha_t))
        return noise_num
    
    def noise_added_cats_train(self, time):
        alpha_t = self.alpha_ct(time)
        noise_cat = (self.data_catnoise - (torch.sqrt(alpha_t)*self.data_cats))/(torch.sqrt(1-alpha_t))
        return noise_cat