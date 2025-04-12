import numpy as np
import pandas as pd
import hashlib

def sha256(val: str, n_buckets:int = 2**32):
    digest = hashlib.sha256(val.encode('utf-8')).digest()
    val_64 = int.from_bytes(digest [:18], byteorder = 'big', signed=False)
    return val_64 % n_buckets

def hash_encode(series: pd.Series, n_buckets:int = 2**32):
    return series.apply(lambda x: sha256(x, n_buckets))
    
def data_review(train:bool = True) -> pd.DataFrame:
    if train == True:
        train_df = pd.read_csv('data/train_data.csv')
    else:
        train_df = pd.read_csv('data/test_data.csv')
    train_df.drop_duplicates(inplace=True)
    train_df.reset_index(inplace=True)  # Drop unwanted columns
    train_df = train_df.drop(columns=['Location', 'Address type', 'Ownership', 'FarmerID'])
    # Filter the target variable column as before
    train_df = train_df[train_df['Target_Variable/Total Income'] != 0]
    # Instead of filtering zero for 'Perc_of_house_with_6plus_room', drop rows missing this value
    train_df = train_df.dropna(subset=['Perc_of_house_with_6plus_room', 'Total_Land_For_Agriculture'])
    
    train_df.loc[train_df['No_of_Active_Loan_In_Bureau'] == 0, 'Avg_Disbursement_Amount_Bureau'] = 0
    
    # Apply feature hashing only on non-numeric categorical columns
    obj_cols = train_df.select_dtypes(include=['object']).columns
    obj_cols = [col for col in obj_cols if col not in ['VILLAGE', 'DISTRICT', 'State', 'Zipcode']]
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
        hashed_values = hash_encode(train_df[col], n_buckets=n_features)       
        
        hashed_dfs.append(hashed_values)
    
    # Combine hashed features with non-object columns
    if hashed_dfs:
        hashed_features_df = pd.concat(hashed_dfs, axis=1)
        train_df = pd.concat([non_obj_df, hashed_features_df], axis=1)
    else:
        train_df = non_obj_df
    
    return train_df

def read_csv_clean(csv_path:str):
    train_df_unpacked = pd.read_csv(csv_path, index_col=0)
    return train_df_unpacked

if __name__ == "__main__":
    train_df = data_review()
    train_df.to_csv('data/train_data_cleaned.csv', index=True)
    train_df_unpacked = read_csv_clean('data/train_data_cleaned.csv')
    print(train_df_unpacked.head())
    print(train_df.head())
    print(train_df.compare(train_df_unpacked))
    print(train_df_unpacked.columns)
    