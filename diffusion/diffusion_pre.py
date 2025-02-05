import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
 
def preprocess(df, nums, cats):
    '''
    Preprocesses data using MinMaxScaler for numericals and OneHotEncoder for categoricals.
    
    Args: df (pd.DataFrame): Dataframe containing both numerical and categorical columns.
        nums (list): List of numerical columns.
        cats (list): List of categorical columns.
    
    Returns: scaler (MinMaxScaler): Fitted MinMaxScaler object.
        encoder (OneHotEncoder): Fitted OneHotEncoder object.
        tensors_num (torch.Tensor): Tensor containing scaled numerical data.
        tensors_cat (torch.Tensor): Tensor containing one-hot encoded categorical data.
    '''
    scaler = MinMaxScaler()
    encoder = OneHotEncoder()
    num_data = scaler.fit_transform(df[nums])
    cat_data = encoder.fit_transform(df[cats]).toarray()  # Ensure it's a 2D array
    tensors_cat = torch.tensor(cat_data, dtype=torch.float32)
    tensors_num = torch.tensor(num_data, dtype=torch.float32)
    num_num_cols = tensors_num.shape[1]
    num_cat_cols = tensors_cat.shape[1]
    # Create mapping for numerical columns
    num_col_mapping = {i: col_name for i, col_name in enumerate(nums)}
    # Create mapping for categorical columns after one-hot encoding
    cat_col_names = encoder.get_feature_names_out(cats)
    cat_col_mapping = {i: col_name for i, col_name in enumerate(cat_col_names)}
    return scaler, encoder, tensors_num, tensors_cat, num_num_cols, num_cat_cols, num_col_mapping, cat_col_mapping

class ForwardDiffuse:
    '''
    Class to perform forward diffusion on numerical and categorical data.
    
    Init_args: data_nums (torch.Tensor): Tensor containing scaled numerical data.
        data_cats (torch.Tensor): Tensor containing one-hot encoded categorical data.
        num_steps (int): Number of diffusion steps.
        total_time (int): Total time for diffusion.
        s (float): Smoothing parameter
        
    Methods: _alpha_ct: Computes alpha_t for a given time.
        _compute_alpha_schedule: Computes alpha_t for all time steps.
        forward_diffuse_num: Adds noise to numerical data.
        forward_diffuse_cats: Adds noise to categorical data.
    '''
    def __init__(self, data_nums, data_cats, num_steps, total_time, s=0.008):
        self.num_steps = num_steps  # Renamed parameter
        self.total_time = total_time
        self.data_nums = data_nums
        self.data_cats = data_cats
        self.s = s
        self.alpha_schedule = self._compute_alpha_schedule()  # Precompute schedule
    
    def _alpha_ct(self, time):
        return (torch.cos((((time / self.total_time) + self.s) / (1 + self.s)) * (torch.pi / 2)))**2
    
    def _compute_alpha_schedule(self):
        schedule = torch.linspace(0, self.total_time, steps=self.num_steps)
        return torch.tensor([self._alpha_ct(t) for t in schedule])
    
    def forward_diffuse_num(self, time_idx, nums):
        alpha_t = self.alpha_schedule[time_idx]
        epsilon = torch.randn_like(nums)
        nums_noise = torch.sqrt(alpha_t) * nums + torch.sqrt(1 - alpha_t) * epsilon
        return nums_noise
    
    def forward_diffuse_cats(self, time_idx, cats):
        alpha_t = self.alpha_schedule[time_idx]
        epsilon = torch.randn_like(cats)
        cats_noise = torch.sqrt(alpha_t) * cats + torch.sqrt(1 - alpha_t) * epsilon
        return cats_noise

def Forward_Diffuse(data_nums, data_cats, num_steps, total_time, s=0.008):
    '''
    Function to perform Forward Diffusion using class Forward Diffusion
    
    Args: data_nums (torch.Tensor): Tensor containing scaled numerical data.
        data_cats (torch.Tensor): Tensor containing one-hot encoded categorical data.
        num_steps (int): Number of diffusion steps.
        total_time (int): Total time for diffusion.
        s (float): Smoothing parameter
        
    Returns: data_tensor (torch.Tensor): Tensor containing noisy numerical and categorical data.
        data (torch.Tensor): Tensor containing original numerical and categorical data.
        time (torch.Tensor): Tensor containing time indices
    '''
    forward_diffuse = ForwardDiffuse(data_nums, data_cats, num_steps, total_time, s)
    batch_size = data_nums.shape[0]
    num_num_cols= data_nums.shape[1]
    num_cat_cols = data_cats.shape[1]
    data_size = data_nums.shape[1] + data_cats.shape[1] + 1
    data_tensor = torch.zeros((num_steps, batch_size, data_size))
    
    for idx in range(num_steps):
        data_nums_noise = forward_diffuse.forward_diffuse_num(idx, data_nums)
        data_cats_noise = forward_diffuse.forward_diffuse_cats(idx, data_cats)
        data_tensor[idx, :, :-1] = torch.cat((data_nums_noise, data_cats_noise), dim=1)
        data_tensor[idx, :, -1] = idx  # Assign timestep index
    
    return data_tensor, torch.cat((data_nums, data_cats), dim=1), torch.arange(num_steps, dtype = torch.long), num_num_cols, num_cat_cols

class DeNoiseData:
    '''
    Class to obtain noise data from noisy data and clean data
    
    Init_args: data_nums_cl (torch.Tensor): Tensor containing clean numerical data.
        data_cats_cl (torch.Tensor): Tensor containing clean categorical data.
        data_nums_n (torch.Tensor): Tensor containing noisy numerical data.
        data_cats_n (torch.Tensor): Tensor containing noisy categorical data.
        num_steps (int): Number of diffusion steps.
        total_time (int): Total time for diffusion.
        s (float): Smoothing parameter
    
    Methods: noise_added_num: Computes noise added to numerical data.
        noise_added_cat: Computes noise added to categorical data.
    '''
    def __init__(self, data_nums_cl, data_cats_cl, data_nums_n, data_cats_n, num_steps, total_time, s=0.008):
        self.num_steps = num_steps  # Renamed parameter
        self.data_nums = data_nums_cl
        self.data_cats = data_cats_cl
        self.timestep = num_steps
        self.total_time = total_time
        self.s = s
        self.data_numnoise = data_nums_n
        self.data_catnoise = data_cats_n
        self.alpha_schedule = self._compute_alpha_schedule()  # Precompute schedule
    
    def _alpha_ct(self, time):
        return (torch.cos((((time / self.total_time) + self.s) / (1 + self.s)) * (torch.pi / 2))) ** 2

    def _compute_alpha_schedule(self):
        schedule = torch.linspace(0, self.total_time, steps=self.num_steps)
        return torch.tensor([self._alpha_ct(t) for t in schedule])

    def noise_added_num(self, time_idx):
        alpha_t = self.alpha_schedule[time_idx]
        noise_num = (self.data_numnoise - (torch.sqrt(alpha_t)*self.data_nums))/(torch.sqrt(1-alpha_t))
        return noise_num
    
    def noise_added_cats(self, time_idx):
        alpha_t = self.alpha_schedule[time_idx]
        noise_cat = (self.data_catnoise - (torch.sqrt(alpha_t)*self.data_cats))/(torch.sqrt(1-alpha_t))
        return noise_cat