import pandas as pd
from maps import request_maps
from data_cleaners.data_cleanup import read_csv_clean
from soilgrids import soilgrids_df

def get_pos()-> pd.DataFrame:
    df = read_csv_clean('train_data_cleaned.csv')
    df = df[['VILLAGE', 'DISTRICT', 'State', 'Zipcode']]
    return df
    
def return_coordinates(df:pd.DataFrame) -> pd.DataFrame:
    lat_lon = pd.DataFrame(columns=['lat', 'lon'], index=df.index)
    for i, row in df.iterrows():
        village = row['VILLAGE']
        district = row['DISTRICT']
        state = row['State']
        zipcode = row['Zipcode']
        lat, lon = request_maps(village, district, state, zipcode)
        lat_lon.loc[i] = [lat, lon]
    return lat_lon

def return_soilgrids(lat_lon:pd.DataFrame) -> pd.DataFrame:
    soil_classes = soilgrids_df(lat_lon)
    return soil_classes

locations_raw = get_pos()
locations = return_coordinates(locations_raw)
soil_classes = return_soilgrids(locations)
soil_classes.to_csv('soil_classes.csv', index=True)