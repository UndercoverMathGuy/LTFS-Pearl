import pandas as pd  # Import pandas for DataFrame operations
import torch
from diffusion_pre import Forward_Diffuse, preprocess
from tabtransformer import train_model, create_data

def train_data(input_data, num_steps, total_time, s=0.008):
    scaler, encoder, input_clean_nums, input_clean_cats, num_num_cols, num_cat_cols, num_col_mapping, cat_col_mapping = preprocess(input_data)
    data_noisy, data_clean, timesteps, num_num_cols, num_cat_cols = Forward_Diffuse(
        data_nums=input_clean_nums,
        data_cats=input_clean_cats,
        num_steps=num_steps,
        total_time=total_time,
        s=s
    )
    train_model(data_noisy, data_clean, timesteps, num_steps, total_time, s=s)
    return scaler, encoder, num_num_cols, num_cat_cols, num_col_mapping, cat_col_mapping  # Return mappings

def reconstruct_dataframe(nums_decoded, cats_decoded, num_col_mapping, cat_col_mapping):
    # Create DataFrames with original column names
    nums_df = pd.DataFrame(nums_decoded, columns=[num_col_mapping[i] for i in range(nums_decoded.shape[1])])
    cats_df = pd.DataFrame(cats_decoded, columns=[cat_col_mapping[i] for i in range(cats_decoded.shape[1])])
    # Combine numerical and categorical data
    decoded_df = pd.concat([nums_df, cats_df], axis=1)
    return decoded_df

def creator(input_sample, total_time_train, num_num_cols, num_cat_cols, scaler, encoder, num_col_mapping, cat_col_mapping):
    noise = torch.randn_like(input_sample)
    nums_reconstructed, cats_reconstructed = create_data(
        noise_data=noise,
        timestep=total_time_train,
        num_num_cols=num_num_cols,
        num_cat_cols=num_cat_cols
    )
    nums_decoded = scaler.inverse_transform(nums_reconstructed.numpy())
    cats_decoded = encoder.inverse_transform(cats_reconstructed.numpy())
    # Use the new function to reconstruct DataFrame
    decoded_df = reconstruct_dataframe(nums_decoded, cats_decoded, num_col_mapping, cat_col_mapping)
    return decoded_df
