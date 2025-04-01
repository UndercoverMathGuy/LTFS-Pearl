import numpy as np
import pandas as pd
import sklearn.preprocessing as skl

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
    
    # Apply one-hot encoding on object/categorical columns using sklearn
    obj_cols = train_df.select_dtypes(include=['object']).columns
    non_obj_df = train_df.drop(columns=obj_cols)
    encoder = skl.OneHotEncoder(drop='first', sparse_output=False)
    encoded_arr = encoder.fit_transform(train_df[obj_cols])
    encoded_df = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out(obj_cols), index=train_df.index)
    train_df = pd.concat([non_obj_df, encoded_df], axis=1)
    
    return train_df

if __name__ == "__main__":
    train_df = data_review()
    print(train_df.head())

