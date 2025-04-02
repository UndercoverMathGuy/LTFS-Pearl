import numpy as np
import pandas as pd
import hashlib

def sha_bucket(category_value: str, n_buckets: int = 2**16) -> int:
    """
    Hashes a string category_value using SHA-256, then maps it to [0, n_buckets-1].
    
    :param category_value: The category string
    :param n_buckets: Number of buckets (must be <= 2^(some bits of truncated hash))
    :return: An integer bucket index in [0, n_buckets-1]
    """
    # 1) Compute SHA-256 digest (32 bytes)
    digest_bytes = hashlib.sha256(category_value.encode('utf-8')).digest()
    # 2) Convert the first 8 bytes (or 16, etc.) to a 64-bit integer
    #    If you need more bits, read more of the digest
    #    For n_buckets up to 2^16 or 2^20, 64 bits is usually enough.
    long_val = int.from_bytes(digest_bytes[:8], byteorder='big', signed=False)
    # 3) Mod by n_buckets
    bucket_idx = long_val % n_buckets
    return bucket_idx

def data_review(train:bool = True) -> pd.DataFrame:
    if train == True:
        train_df = pd.read_csv('data/train_data.csv')
    else:
        train_df = pd.read_csv('data/test_data.csv')
    train_df.drop_duplicates(inplace=True)
    train_df.reset_index(inplace=True)  # Drop unwanted columns
    train_df = train_df.drop(columns=['Location', 'Address type', 'Ownership', 'FarmerID', 'Zipcode'])
    # Filter the target variable column as before
    train_df = train_df[train_df['Target_Variable/Total Income'] != 0]
    # Instead of filtering zero for 'Perc_of_house_with_6plus_room', drop rows missing this value
    train_df = train_df.dropna(subset=['Perc_of_house_with_6plus_room', 'Total_Land_For_Agriculture'])
    
    train_df.loc[train_df['No_of_Active_Loan_In_Bureau'] == 0, 'Avg_Disbursement_Amount_Bureau'] = 0
    
    # Apply feature hashing only on non-numeric categorical columns
    obj_cols = train_df.select_dtypes(include=['object']).columns
    
    # Identify numeric categorical columns
    numeric_categorical_cols = []
    for col in obj_cols:
        # Check if the column contains only numeric values
        if pd.to_numeric(train_df[col], errors='coerce').notna().all():
            numeric_categorical_cols.append(col)
            # Convert numeric categorical columns to appropriate numeric type
            train_df[col] = pd.to_numeric(train_df[col])
    
    # Filter out numeric categorical columns from object columns list
    obj_cols = [col for col in obj_cols if col not in numeric_categorical_cols]
    
    # Keep all non-categorical columns and numeric categorical columns
    non_obj_df = train_df.drop(columns=obj_cols)
    
    # Process each non-numeric categorical column with SHA-256 hashing
    hashed_dfs = []
    n_features = 256
    
    for col in obj_cols:
        # Apply SHA-256 hashing to this column
        hashed_values = train_df[col].astype(str).apply(lambda x: sha_bucket(x, n_buckets=n_features))
        
        # Create one-hot encoding for the hashed values
        one_hot_matrix = np.zeros((len(hashed_values), n_features))
        for i, bucket_idx in enumerate(hashed_values):
            one_hot_matrix[i, bucket_idx] = 1
        
        # Create DataFrame with hashed features
        col_prefix = f"{col}_hash_"
        feature_names = [f"{col_prefix}{i}" for i in range(n_features)]
        hashed_df = pd.DataFrame(one_hot_matrix, columns=feature_names, index=train_df.index)
        hashed_dfs.append(hashed_df)
    
    # Combine hashed features with non-object columns
    if hashed_dfs:
        hashed_features_df = pd.concat(hashed_dfs, axis=1)
        train_df = pd.concat([non_obj_df, hashed_features_df], axis=1)
    else:
        train_df = non_obj_df
    
    return train_df



if __name__ == "__main__":
    train_df = data_review()
    print(train_df.head())
    print(len(train_df.columns))
    train_raw = pd.read_csv('data/train_data.csv')
    print(len(train_raw.columns))
