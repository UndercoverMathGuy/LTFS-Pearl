import requests as r
import json
import pandas as pd

def get_soilgrids(lat, lon):
    url = f'https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_class=5'
    response = r.get(url)
    data = json.loads(response.text)
    return data['wrb_class_name']

def soilgrids_df(coords: pd.DataFrame) -> pd.DataFrame:
    # Create an empty DataFrame with the same index as coords
    out = pd.DataFrame(index=coords.index, columns=['class'])
    for idx, row in coords.iterrows():
        lat, lon = row['lat'], row['lon']
        out.at[idx, 'class'] = get_soilgrids(lat, lon)
    return out