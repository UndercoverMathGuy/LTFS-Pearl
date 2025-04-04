from maps import request_maps
from soilgrids import soilgrids_df
import pandas as pd
from data_cleaners.data_cleanup import read_csv_clean

def soilgrids_df_append():
    original_df = read_csv_clean('data/train_data_cleaned.csv')
    soilgrids_df = pd.DataFrame(columns=['soil_type'], index=original_df.index)
    print(original_df['VILLAGE'][1])
    # for i in range(len(original_df)):
    #     latlon = request_maps(village = original_df['VILLAGE'][i], district=original_df['DISTRICT'][i], state=original_df['State'][i], zipcode=original_df['Zipcode'][i])
