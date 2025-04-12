import requests as r
import json
import pandas as pd

def get_soilgrids(lat, lon):
    url = f'https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_class=5'
    response = r.get(url)
    data = json.loads(response.text)
    return data['wrb_class_name']

def soilgrids_df(coords: pd.DataFrame) -> pd.DataFrame:
    soil_classes = pd.DataFrame(columns=['class'])
    for idx, row in coords.iterrows():
        lat = row['lat']
        lon = row['lon']
        soil_class = get_soilgrids(lat, lon)
        df_row = pd.DataFrame({'class': soil_class}, index=idx)
        soil_classes = pd.concat([soil_classes, df_row], ignore_index=False)
    return soil_classes